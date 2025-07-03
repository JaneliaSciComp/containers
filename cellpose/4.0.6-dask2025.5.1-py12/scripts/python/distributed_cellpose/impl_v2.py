"""
This package uses Chris Roat's approach to merge labels
added to cellpose in https://github.com/MouseLand/cellpose/pull/356
"""
import dask
import dask.array as da
import functools
import logging
import numpy as np
import os
import scipy
import skimage
import time
import torch
import traceback

import io_utils.read_utils as read_utils
import io_utils.zarr_utils as zarr_utils

from cellpose import transforms
from dask.distributed import as_completed
from sklearn import metrics as sk_metrics

from .block_utils import (get_block_crops, get_nblocks,
                          compute_block_anisotropy, remove_overlaps)


logger = logging.getLogger(__name__)


def distributed_eval(
        image_container_path,
        image_subpath,
        image_shape,
        timeindex,
        image_channels,
        model_type,
        blocksize,
        output_dir,
        dask_client,
        blocksoverlap=(),
        mask=None,
        preprocessing_steps=[],
        diameter=None,
        do_3D=True,
        z_axis=0,
        channel_axis=None,
        anisotropy=None,
        min_size=15,
        normalize=True,
        normalize_lowhigh=None,
        normalize_percentile=None,
        normalize_norm3D=True,
        normalize_sharpen_radius=0,
        normalize_smooth_radius=0,
        normalize_tile_norm_blocksize=0,
        normalize_tile_norm_smooth3D=1,
        normalize_invert=False,
        flow_threshold=0.4,
        cellprob_threshold=0,
        stitch_threshold=0,
        flow3D_smooth=0,
        iou_threshold=0,
        iou_depth=1,
        label_dist_th=1.0,
        persist_labeled_blocks=False,
        use_gpu=False,
        gpu_device=None,
        gpu_batch_size=8,
        test_mode=False,
):
    """
    partition a 3-D volume into overlapping blocks and run cellpose segmentation
    and distribute the segmentation for each block on a dask cluster
    """
    start_time = time.time()
    image_ndim = len(image_shape)
    # check blocksoverlap
    if blocksoverlap is None or blocksoverlap == ():
        blocksoverlap = [int(s * 0.1) for s in blocksize]
    elif isinstance(blocksoverlap, (int, float)):
        blocksoverlap = [int(blocksoverlap)] * image_ndim
    else:
        raise ValueError((
            f'Invalid blocksoverlap {blocksoverlap} of type {type(blocksoverlap)} '
            f'- expected tuple with same size as image dimensions: {image_ndim}'
        ))

    blockchunks = np.array(blocksize, dtype=int)
    blockoverlaps = np.array(blocksoverlap, dtype=int)

    # extra check in case blocksize and diameter are very close
    for ax in range(len(blockchunks)):
        if blockoverlaps[ax] > blockchunks[ax] / 2:
            blockoverlaps[ax] = int(blockchunks[ax] / 2)

    nblocks = get_nblocks(image_shape, blockchunks)
    logger.info((
        f'Blocksize:{blockchunks}, '
        f'overlap:{blockoverlaps} => {nblocks} blocks '
    ))

    blocks_info = [bi for bi in zip(*get_block_crops(image_shape, blockchunks, blockoverlaps, mask)) ]

    eval_block = functools.partial(
        _eval_model,
        model_type=model_type,
        diameter=diameter,
        min_size=min_size,
        anisotropy=anisotropy,
        do_3D=do_3D,
        z_axis=z_axis,
        channel_axis=channel_axis,
        normalize=normalize,
        normalize_lowhigh=normalize_lowhigh,
        normalize_percentile=normalize_percentile,
        normalize_norm3D=normalize_norm3D,
        normalize_sharpen_radius=normalize_sharpen_radius,
        normalize_smooth_radius=normalize_smooth_radius,
        normalize_tile_norm_blocksize=normalize_tile_norm_blocksize,
        normalize_tile_norm_smooth3D=normalize_tile_norm_smooth3D,
        normalize_invert=normalize_invert,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        flow3D_smooth=flow3D_smooth,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_batch_size=gpu_batch_size,
    )

    segment_block = functools.partial(
        _segment_block,
        eval_block,
        image_container_path=image_container_path,
        image_subpath=image_subpath,
        data_timeindex=timeindex,
        data_channels=image_channels,
        do_3D=do_3D,
        channel_axis=channel_axis,
        blocksize=blockchunks,
        blockoverlaps=blockoverlaps,
        preprocessing_steps=preprocessing_steps,
    )

    logger.info(f'Start segmenting: {len(blocks_info)} {blocksize} blocks with overlap {blocksoverlap}')

    segment_block_res = dask_client.map(
        segment_block,
        blocks_info,
    )

    if (do_3D and len(image_shape) > 3 or
        not do_3D and len(image_shape) > 2):
        labels_shape = [s for i, s in enumerate(image_shape) if i != channel_axis]
        labels_blocksize = [s for i, s in enumerate(blocksize) if i != channel_axis]
    else:
        labels_shape = image_shape
        labels_blocksize = blocksize

    labeled_blocks, labeled_blocks_info, max_label = _collect_labeled_blocks(
        segment_block_res,
        labels_shape,
        labels_blocksize,
        output_dir=output_dir,
        persist_labeled_blocks=persist_labeled_blocks,
    )

    logger.info((
        f'Finished segmentation of {len(blocks_info)} blocks '
        f'in {time.time()-start_time}s '
    ))

    if test_mode:
        # return the labeled blocks before merging the labels
        logger.info('Return labeled blocks')
        return labeled_blocks, []

    if np.prod(nblocks) > 1:
        working_labeled_blocks = labeled_blocks
        logger.info(f'Submit link labels for {nblocks} label blocks')
        labeled_blocks_index_and_coords = [(bi[0], bi[1])
                                           for bi in labeled_blocks_info]
        new_labeling = _link_labels(
            working_labeled_blocks,
            labeled_blocks_index_and_coords,
            max_label,
            iou_depth,
            iou_threshold,
            dask_client,
        )
        # save labels to a temporary file for the relabeling process
        new_labeling_filename = f'{output_dir}/new_labeling.npy'
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Save new labeling to {new_labeling_filename}')
        persisted_new_labeling = dask.delayed(_save_labels)(
            new_labeling,
            new_labeling_filename,
        )
        logger.info(f'Relabel {nblocks} blocks from {persisted_new_labeling}')
        relabeled = da.map_blocks(
            _relabel_block,
            labeled_blocks,
            labels_filename=persisted_new_labeling,
            dtype=labeled_blocks.dtype,
            chunks=labeled_blocks.chunks)
    else:
        logger.info('There is only one block - no link labels is needed')
        relabeled = labeled_blocks
    # return the labels and an empty list for bounding boxes
    return relabeled, []


def _read_block_data(block_info, image_container_path, image_subpath=None,
                     data_timeindex=None, data_channels = None):
    block_index, block_coords = block_info
    logger.info((
        f'Get block: {block_index} at {block_coords} from '
        f'{image_container_path}:{image_subpath}:{data_timeindex}:{data_channels} '
    ))
    block_data, block_attrs = read_utils.open(image_container_path, image_subpath,
                                              data_timeindex=data_timeindex,
                                              data_channels=data_channels,
                                              block_coords=block_coords)
    logger.info(f'Retrieved block {block_index} of shape {block_data.shape} with {block_attrs}')
    return block_data, block_attrs


def _segment_block(eval_method,
                   block_info,
                   image_container_path=None,
                   image_subpath=None,
                   data_timeindex=None,
                   data_channels=None,
                   do_3D = True,
                   channel_axis=None,
                   blocksize=[],
                   blockoverlaps=[],
                   preprocessing_steps=[],
                   ):
    block_index, block_coords = block_info
    block_shape = tuple([sl.stop-sl.start for sl in block_coords])

    start_time = time.time()
    logger.info((
        f'Segment block: {block_index}, {block_coords} '
        f'blocksize: {blocksize}, blockoverlaps: {blockoverlaps} '
        f'block shape: {block_shape} '
    ))
    block_data, block_attrs = _read_block_data(block_info, image_container_path,
                                               image_subpath=image_subpath,
                                               data_timeindex=data_timeindex,
                                               data_channels=data_channels)
    # preprocess
    for pp_step in preprocessing_steps:
        logger.debug(f'Apply preprocessing step: {pp_step}')
        block_data = pp_step[0](block_data, **pp_step[1])

    labels = eval_method(block_index, block_data, block_attrs)

    if (do_3D and len(block_shape) > 3 or
        not do_3D and len(block_shape) > 2):
        # labels are single channel so if the input was multichannel remove the channel coords
        labels_index = [b for i, b in enumerate(block_index) if i != channel_axis]
        labels_coords = [c for i, c in enumerate(block_coords) if i != channel_axis]
        labels_overlaps = [o for i, o in enumerate(blockoverlaps) if i != channel_axis]
        labels_blocksize = [s for i, s in enumerate(blocksize) if i != channel_axis]
    else:
        labels_index = block_index
        labels_coords = block_coords
        labels_overlaps = blockoverlaps
        labels_blocksize = blocksize

    unique_labels_with_overlaps = np.unique(labels)
    labels, labels_coords = remove_overlaps(labels, labels_coords, labels_overlaps, labels_blocksize)
    end_time = time.time()
    # after removing the overlaps some labels may be missing
    # so we need to relabel the block again with contiguous labels
    unique_labels = np.unique(labels)
    logger.info((
        f'Finished segmenting block {block_index} '
        f'in {end_time-start_time}s, '
        f'before cropping there were {len(unique_labels_with_overlaps)} labels, '
        f'after cropping there are {len(unique_labels)} unique labels.'
    ))

    if len(unique_labels) != len(unique_labels_with_overlaps):
        # some labels were only in the overlaps so we need to relabel
        if unique_labels[0] == 0:
            old_labeling = unique_labels
            new_labeling = np.arange(len(unique_labels)).astype(np.uint32)
        else:
            # no background label was present in this block
            # so we need to add it
            old_labeling = np.concatenate(([0], unique_labels))
            new_labeling = np.arange(len(unique_labels) + 1).astype(np.uint32)

        relabeled_block = skimage.util.map_array(labels, old_labeling, new_labeling)

        max_label = np.max(relabeled_block)
        logger.info((
            f'Relabeled block {labels_index} - {relabeled_block.shape} '
            f'block max label: {max_label} '
        ))
        return labels_index, tuple(labels_coords), max_label, relabeled_block
    else:
        max_label = np.max(labels)
        logger.info((
            f'No need to relabel block {labels_index} - {labels.shape} '
            f'block max label: {max_label} '
        ))
        return labels_index, tuple(labels_coords), max_label, labels


def _eval_model(block_index,
                block_data,
                block_attrs,
                model_type=None,
                diameter=None,
                min_size=15,
                anisotropy=None,
                do_3D=True,
                z_axis=0,
                channel_axis=None,
                normalize=True,
                normalize_lowhigh=None,
                normalize_percentile=None,
                normalize_norm3D=True,
                normalize_sharpen_radius=0,
                normalize_smooth_radius=0,
                normalize_tile_norm_blocksize=0,
                normalize_tile_norm_smooth3D=1,
                normalize_invert=False,
                flow_threshold=0.4,
                cellprob_threshold=0,
                stitch_threshold=0,
                flow3D_smooth=0,
                use_gpu=False,
                gpu_device=None,
                gpu_batch_size=8,
                ):
    from cellpose import models

    logger.info((
        f'Run model eval for block: {block_index}, '
        f'size: {block_data.shape}, '
        f'model_type: {model_type} '
    ))

    np.random.seed(block_index)

    start_time = time.time()
    if use_gpu:
        available_gpus = torch.cuda.device_count()
        logger.info(f'Found {available_gpus} GPUs')
        if available_gpus > 0:
            # if multiple gpus are available try to find one that can be used
            segmentation_device, gpu = None, False
            for gpui in range(available_gpus):
                try:
                    logger.debug(f'Try GPU: {gpui}')
                    segmentation_device, gpu = models.assign_device(gpu=use_gpu,
                                                                    device=gpui)
                    logger.debug(f'Result for GPU: {gpui} => {segmentation_device}:{gpu}')
                    if gpu:
                        break
                    # because of a bug in cellpose trying the other devices explicitly here
                    torch.cuda.set_device(gpui)
                    segmentation_device = torch.device(f'cuda:{gpui}')
                    logger.info(f'Device {segmentation_device} present and usable')
                    _ = torch.zeros((1,1)).to(segmentation_device)
                    logger.info(f'Device {segmentation_device} tested and it is usable')
                    gpu = True
                    break
                except Exception as e:
                    logger.warning(f'cuda:{gpui} present but not usable: {e}')
        else:
            segmentation_device, gpu = models.assign_device(gpu=use_gpu,
                                                            device=gpu_device)
    else:
        segmentation_device, gpu = models.assign_device(gpu=use_gpu,
                                                        device=gpu_device)
    logger.info(f'Segmentation device for block {block_index}: {segmentation_device}:{gpu}')
    model = models.CellposeModel(gpu=gpu,
                                 pretrained_model=model_type,
                                 device=segmentation_device)
    normalize_params = {
        "normalize": normalize,
        "lowhigh": ((int(normalize_lowhigh[0]), int(normalize_lowhigh[1]))
                       if normalize_lowhigh is not None else None),
        "percentile": ((int(normalize_percentile[0]), int(normalize_percentile[1]))
                       if normalize_percentile is not None else None),
        "norm3D": normalize_norm3D,
        "sharpen_radius": normalize_sharpen_radius,
        "smooth_radius": normalize_smooth_radius,
        "tile_norm_blocksize": normalize_tile_norm_blocksize,
        "tile_norm_smooth3D": normalize_tile_norm_smooth3D,
        "invert": normalize_invert,
    }
    if (do_3D and len(block_data.shape) == 3 or
        not do_3D and len(block_data.shape) == 2):
        # if 3D and the block has exactly 3 dimensions
        # or in the case of 2D segmentation the block has exactly 2 dimensions
        # reshape it to include a dimension for the channel
        new_block_shape = (1,) + block_data.shape
        logger.debug(f'Reshape block of {block_data.shape} to {new_block_shape}')
        block_data = block_data.reshape(new_block_shape)

    if normalize:
        logger.info(f'Normalize params: {normalize_params}')
        block_data = transforms.normalize_img(block_data, axis=channel_axis,
                                              **normalize_params)

    if anisotropy is None or anisotropy == 1:
        # try to compute it from block's attributes
        anisotropy = compute_block_anisotropy(block_attrs)

    logger.info((
        'Eval args: '
        f'diameter={diameter}, '
        f'min_size={min_size}, '
        f'anisotropy={anisotropy}, '
        f'do_3D={do_3D}, '
        f'z_axis={z_axis}, '
        f'normalize=False, '
        f'channel_axis={channel_axis}, '
        f'flow_threshold={flow_threshold}, '
        f'cellprob_threshold={cellprob_threshold}, '
        f'stitch_threshold={stitch_threshold}, '
        f'flow3D_smooth={flow3D_smooth}, '
        f'batch_size={gpu_batch_size}, '
    ))
    labels = model.eval(block_data,
                        diameter=diameter,
                        min_size=min_size,
                        anisotropy=anisotropy,
                        do_3D=do_3D,
                        z_axis=z_axis,
                        normalize=False,
                        channel_axis=channel_axis,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        stitch_threshold=stitch_threshold,
                        flow3D_smooth=flow3D_smooth,
                        batch_size=gpu_batch_size,
                        )[0]
    end_time = time.time()
    unique_labels = np.unique(labels)
    logger.info((
        f'Finished model eval for block: {block_index} '
        f'found {len(unique_labels)} unique labels '
        f'in {end_time-start_time}s '
    ))
    return labels.astype(np.uint32)


def _collect_labeled_blocks(segment_blocks_res, shape, chunksize,
                            output_dir=None,
                            persist_labeled_blocks=False):
    """
    Collect segmentation results.

    Parameters
    ==========
    segment_blocks_res: block segmentation results
    shape: shape of a full image being segmented
    chunksize: result array chunksize

    Returns
    =======
    labels: dask array created from segmentation results

    """
    logger.info('Begin collecting labeled blocks (blocksize={chunksize})')
    labeled_blocks_info = []
    if output_dir is not None and persist_labeled_blocks:
        # collect labels in a zarr array
        labeled_blocks_zarr_container = f'{output_dir}/labeled_blocks.zarr'
        logger.info((
            'Save labels to temporary zarr ' 
            f'{labeled_blocks_zarr_container} '
        ))
        labels = zarr_utils.create_dataset(
            labeled_blocks_zarr_container,
            '',
            shape,
            chunksize,
            np.uint32,
            data_store_name='zarr',
        )
        is_zarr = True
    else:
        labels = da.empty(shape, dtype=np.uint32, chunks=chunksize)
        is_zarr = False
    result_index = 0
    max_label = 0
    # collect segmentation results
    completed_segment_blocks = as_completed(segment_blocks_res,
                                            with_results=True)
    for f, r in completed_segment_blocks:
        logger.debug(f'Process future {f}')
        if f.cancelled():
            exc = f.exception()
            tb = f.traceback()
            logger.error((
                f'Process future {f} error: {exc} '
                f'{traceback.extract_tb(tb)} '
            ))

        (block_index, block_coords, max_block_label, block_labels) = r
        block_shape = tuple([sl.stop-sl.start for sl in block_coords])

        logger.info((
            f'{result_index+1}. '
            f'Write labels {block_index},{block_coords} '
            f'block shape: {block_shape} ?==? {block_labels.shape} '
            f'max block label: {max_block_label} '
            f'block labels range: {max_label} - {max_label+max_block_label} '
        ))

        block_labels_offsets = np.where(block_labels > 0,
                                        max_label,
                                        np.uint32(0)).astype(np.uint32)
        block_labels += block_labels_offsets
        # set the block in the dask array of labeled blocks
        labels[block_coords] = block_labels
        # block_index, block_coords, labels_range
        labeled_blocks_info.append((block_index,
                                    block_coords,
                                    (max_label, max_label+max_block_label)))
        max_label = max_label + max_block_label
        result_index += 1

    logger.info(f'Finished collecting labels in {shape} image')
    labels_res = da.from_zarr(labels) if is_zarr else labels
    return labels_res, labeled_blocks_info, max_label


def _link_labels(labels, blocks_index_and_coords, max_label, face_depth,
                 iou_threshold, client):
    label_groups = _get_adjacent_label_mappings(labels,
                                                blocks_index_and_coords,
                                                face_depth,
                                                iou_threshold,
                                                client)
    logger.debug((
        f'Find connected components for {label_groups.shape} label groups: '
        f'max label: {max_label}, label groups: {label_groups} '
    ))
    return dask.delayed(_get_labels_connected_comps)(label_groups, max_label+1)


def _get_adjacent_label_mappings(labels, blocks_index_and_coords,
                                 block_face_depth, iou_threshold,
                                 client):
    logger.debug(f'Create adjacency graph for {labels}')
    blocks_faces_by_axes = _get_blocks_faces_info(blocks_index_and_coords,
                                                  block_face_depth,
                                                  labels)
    logger.debug(f'Invoke label mapping for {len(blocks_faces_by_axes)} faces')
    mapped_labels = client.map(
        _across_block_label_grouping,
        blocks_faces_by_axes,
        iou_threshold=iou_threshold,
        image=labels
    )
    logger.debug('Start collecting label mappings')
    all_mappings = [np.empty((2, 0), dtype=labels.dtype)]
    completed_mapped_labels = as_completed(mapped_labels, with_results=True)
    for _, mapped in completed_mapped_labels:
        logger.debug(f'Append mapping: {mapped}')
        all_mappings.append(mapped)

    mappings = np.concatenate(all_mappings, axis=1)
    logger.debug((
        f'Concatenated {len(all_mappings)} mappings -> '
        f'{mappings.shape}. '
        f'Label mappings: {mappings} '
    ))
    return mappings


def _get_blocks_faces_info(blocks_index_and_coords, face_depth, image):
    ndim = image.ndim
    image_shape = image.shape
    depth = da.overlap.coerce_depth(ndim, face_depth)
    face_slices_and_axes = []
    for bi_and_coords in blocks_index_and_coords:
        block_index, block_coords = bi_and_coords
        logger.debug(f'Get faces for: {block_index}:{block_coords}')
        block_faces = []
        for ax in range(ndim):
            if block_coords[ax].stop >= image_shape[ax]:
                # end block on {ax}
                continue
            block_face = tuple(
                [s if si != ax
                 else slice(block_coords[ax].stop - depth[ax],
                            block_coords[ax].stop + depth[ax])
                 for si, s in enumerate(block_coords)])
            block_faces.append((ax, block_face))
        logger.debug(f'Block: {block_index} - add {len(block_faces)}: {block_faces}')
        face_slices_and_axes.extend(block_faces)
    logger.debug((
        f'There are {len(face_slices_and_axes)} '
        f'face slices to map: {face_slices_and_axes} '
    ))
    return face_slices_and_axes


def _across_block_label_grouping(face_info, iou_threshold=0, image=None):
    axis, face_slice = face_info
    face_shape = tuple([s.stop-s.start for s in face_slice])
    logger.debug((
        f'Group labels for face {face_slice} ({face_shape}) '
        f'along {axis} axis '
    ))
    face = image[face_slice].compute()
    logger.debug(f'Label grouping accross axis {axis} for {face_slice} image')
    unique, unique_indexes = np.unique(
        face, return_index=True,
    )
    logger.debug((
        f'Unique labels for face {face_slice} ({face_shape}) '
        f'along {axis} axis '
        f'{unique}, {unique_indexes} '
    ))
    face0, face1 = np.split(face, 2, axis)
    face0_unique, face0_unique_indexes, face0_unique_counts = np.unique(
        face0.reshape(-1), return_index=True, return_counts=True,
    )
    face1_unique, face1_unique_indexes, face1_unique_counts = np.unique(
        face1.reshape(-1), return_index=True, return_counts=True,
    )
    logger.debug((
        f'Unique labels for face0 of {face_slice} with shape: {face0.shape} '
        f'{face0_unique}, {face0_unique_indexes}, {face0_unique_counts} '
    ))
    logger.debug((
        f'Unique labels for face1 of {face_slice} with shape: {face1.shape} '
        f'{face1_unique}, {face1_unique_indexes}, {face1_unique_counts} '
    ))

    intersection = sk_metrics.confusion_matrix(
        face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)
    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection
    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0)
    logger.debug((
        f'Face intersection for {face_slice} '
        f'sum0: {sum0}, sum1:{sum1.transpose()} '
        f'iou: {iou} '
    ))

    labels0, labels1 = np.nonzero(iou > iou_threshold)
    logger.debug(f'labels0 for {face_slice}: {labels0}')
    logger.debug(f'labels1 for {face_slice}: {labels1}')

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    logger.debug(f'orig labels0 for {face_slice}: {labels0_orig}')
    logger.debug(f'orig labels1 for {face_slice}: {labels1_orig}')
    if labels1_orig.max() < labels0_orig.max():
        grouped = np.stack([labels1_orig, labels0_orig])
    else:
        grouped = np.stack([labels0_orig, labels1_orig])
    logger.debug(f'Current labels for {face_slice}: {grouped}')
    # Discard any mappings with bg pixels
    valid = np.all(grouped != 0, axis=0)
    # if there's not more than one label return it as is
    label_mapping = grouped[:, valid]
    logger.debug((
        f'Valid labels for {face_slice}: '
        f'label mapping: {label_mapping} '
    ))
    return label_mapping


def _get_labels_connected_comps(label_groups, nlabels):
    # reformat label mappings as csr_matrix
    csr_label_groups = _mappings_as_csr(label_groups, nlabels+1)

    logger.debug((
        f'Build connected components for {csr_label_groups.shape} label groups'
        f'{csr_label_groups}'
    ))
    n_comps, connected_comps = scipy.sparse.csgraph.connected_components(
        csr_label_groups,
        directed=False,
    )
    logger.debug((
        f'Connected components: {n_comps}, '
        f'Connected labels: {connected_comps.shape} -> {connected_comps}'
    ))
    return connected_comps


def _mappings_as_csr(lmapping, n):
    logger.debug(f'Generate csr matrix for {lmapping.shape} labels')
    l0 = lmapping[0, :]
    l1 = lmapping[1, :]
    v = np.ones_like(l0)
    mat = scipy.sparse.coo_matrix((v, (l0, l1)), shape=(n, n))
    csr_mat = mat.tocsr()
    logger.debug(f'Labels mapping as csr matrix {csr_mat}')
    return csr_mat


def _save_labels(values, lfilename):
    np.save(lfilename, values)
    return lfilename


def _relabel_block(block,
                   labels_filename=None,
                   block_info=None):
    if block_info is not None and labels_filename is not None:
        logger.debug(f'Relabeling block {block_info[0]}')
        labels = np.load(labels_filename)
        relabeled_block = labels[block]
        return relabeled_block
    else:
        return block
