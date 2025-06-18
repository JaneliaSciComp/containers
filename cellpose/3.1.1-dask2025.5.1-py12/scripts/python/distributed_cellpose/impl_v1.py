"""
This is code contributed by Greg Fleishman to run Cellpose on a Dask cluster.
"""
import cellpose.models
import dask.array as da
import dask_image.ndmeasure as di_ndmeasure
import logging
import numpy as np
import scipy
import time
import torch

import io_utils.read_utils as read_utils
import io_utils.zarr_utils as zarr_utils

from cellpose import transforms

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
        diameter,
        blocksize,
        output_dir,
        dask_client,
        blocksoverlap=(),
        mask=None,
        preprocessing_steps=[],
        use_gpu=False,
        gpu_device=None,
        eval_channels=None,
        do_3D=True,
        normalize=True,
        normalize_lowhigh=None,
        normalize_percentile=None,
        normalize_norm3D=True,
        normalize_sharpen_radius=0,
        normalize_smooth_radius=0,
        normalize_tile_norm_blocksize=0,
        normalize_tile_norm_smooth3D=1,
        normalize_invert=False,
        z_axis=0,
        channel_axis=None,
        anisotropy=None,
        min_size=15,
        resample=True,
        flow_threshold=0.4,
        cellprob_threshold=0,
        stitch_threshold=0,
        gpu_batch_size=8,
        iou_depth=1,
        iou_threshold=0,
        label_dist_th=1.0,
        persist_labeled_blocks=True,
        test_mode=False,
):
    """
    Evaluate a cellpose model on overlapping blocks of a big image.
    Distributed over workstation or cluster resources with Dask.
    Optionally run preprocessing steps on the blocks before running cellpose.
    Optionally use a mask to ignore background regions in image.

    The dask client must be present but it can be either a remote client that references
    a Dask Scheduler's IP or a local client.

    Parameters
    ----------
    image_container_path : string
        Path to the image data container. Supported formats are .nrrd, .tif, .tiff, .npy, .n5, and .zarr.

    image_subpath : string
        Dataset path relative to path.

    image_shape : tuple
        Image shape in voxels. E.g. (512, 1024, 1024)

    timeindex : string
        if the image is a 5-D TCZYX ndarray specify which timeindex to use

    image_channels : sequence[int] | None
        if the image is a multichannel image specify which channels to use
                
    blocksize : iterable
        The size of blocks in voxels. E.g. [128, 256, 256]

    dask_client : dask.distributed.Client
        A remote or locakl dask client.

        mask : numpy.ndarray (default: None)
        A foreground mask for the image data; may be at a different resolution
        (e.g. lower) than the image data. If given, only blocks that contain
        foreground will be processed. This can save considerable time and
        expense. It is assumed that the domain of the input_zarr image data
        and the mask is the same in physical units, but they may be on
        different sampling/voxel grids.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image blocks before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    Returns
    -------
    Two values are returned:
    (1) A reference to the zarr array on disk containing the stitched cellpose
        segments for your entire image
    (2) Bounding boxes for every segment. This is a list of tuples of slices:
        [(slice(z1, z2), slice(y1, y2), slice(x1, x2)), ...]
        The list is sorted according to segment ID. That is the smallest segment
        ID is the first tuple in the list, the largest segment ID is the last
        tuple in the list.
    """
    if diameter <= 0:
        # always specify the diameter
        diameter = 30
    image_ndim = len(image_shape)
    logger.info((
        f'Segment {image_container_path}:{image_subpath} '
        f'shape: {image_shape}, '
        f'process blocks: {blocksize} '
        f'timeindex: {timeindex} '
        f'image channels {image_channels} '
    ))
    # check blocksoverlap
    if blocksoverlap is None:
        blocksoverlap = (int(diameter*2),) * image_ndim
    elif isinstance(blocksoverlap, (int, float)):
        blocksoverlap = (int(blocksoverlap),) * image_ndim
    elif isinstance(blocksoverlap, tuple):
        if len(blocksoverlap) < image_ndim:
            blocksoverlap = blocksoverlap + (int(diameter*2),) * (image_ndim-len(blocksoverlap))
    else:
        raise ValueError(f'Invalid blocksoverlap {blocksoverlap} of type {type(blocksoverlap)} - expected tuple')

    blocksoverlap_arr = np.array(blocksoverlap, dtype=int)
    block_indices, block_crops = get_block_crops(
        image_shape, blocksize, blocksoverlap_arr, mask,
    )

    if (do_3D and len(image_shape) > 3 or 
        not do_3D and len(image_shape) > 2):
        segmentation_shape = [s for i, s in enumerate(image_shape) if i != channel_axis]
        segmentation_block = [s for i, s in enumerate(blocksize) if i != channel_axis]
    else:
        segmentation_shape = image_shape
        segmentation_block = blocksize

    segmentation_zarr_path = f'{output_dir}/segmentation.zarr'
    logger.info((
        f'Create temporary {segmentation_shape} labels '
        f'at {segmentation_zarr_path} with {segmentation_block} chunks'
    ))

    labels_zarr = zarr_utils.create_dataset(
        segmentation_zarr_path,
        'block_labels',
        segmentation_shape,
        segmentation_block,
        np.uint32,
        data_store_name='zarr',
    )

    logger.info(
        f'Start segmenting: ({len(block_indices)}, {len(block_crops)}) '
        f'{blocksize} blocks with overlap {blocksoverlap}')
    futures = dask_client.map(
        process_block,
        block_indices,
        block_crops,
        image_container_path=image_container_path,
        image_subpath=image_subpath,
        image_shape=image_shape,
        data_timeindex=timeindex,
        data_channels=image_channels,
        blocksize=blocksize,
        blocksoverlap=blocksoverlap_arr,
        labels_output_zarr=labels_zarr,
        preprocessing_steps=preprocessing_steps,
        model_type=model_type,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        diameter=diameter,
        eval_channels=eval_channels,
        do_3D=do_3D,
        normalize=normalize,
        normalize_lowhigh=normalize_lowhigh,
        normalize_percentile=normalize_percentile,
        normalize_norm3D=normalize_norm3D,
        normalize_sharpen_radius=normalize_sharpen_radius,
        normalize_smooth_radius=normalize_smooth_radius,
        normalize_tile_norm_blocksize=normalize_tile_norm_blocksize,
        normalize_tile_norm_smooth3D=normalize_tile_norm_smooth3D,
        normalize_invert=normalize_invert,
        z_axis=z_axis,
        channel_axis=channel_axis,
        anisotropy=anisotropy,
        min_size=min_size,
        resample=resample,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        gpu_batch_size=gpu_batch_size,
        test_mode=test_mode,
    )

    results = dask_client.gather(futures)
    logger.info((
        f'Finished segmenting: {len(block_indices)} {blocksize} blocks '
        f'with overlap {blocksoverlap}'
        ' - start label merge process'
    ))

    faces, boxes_, box_ids_ = list(zip(*results))
    logger.info((
        'Segmentation results contain '
        f'faces: {len(faces)}, boxes: {len(boxes_)}, box_ids: {len(box_ids_)}'
    ))

    segmentation_da = da.from_zarr(labels_zarr)

    if test_mode:
        # return the labeled blocks before merging the labels
        logger.info('Return labeled blocks')
        return segmentation_da, []

    boxes = [box for sublist in boxes_ for box in sublist]
    box_ids = np.concatenate(box_ids_).astype(np.uint32)
    logger.info((
        f'Relabel {box_ids.shape} blocks of type {box_ids.dtype} - '
        f'use {len(faces)} faces for merging labels'
    ))
    if (do_3D and len(image_shape) > 3 or 
        not do_3D and len(image_shape) > 2):
        label_block_indices = []
        for bi in block_indices:
            label_block_indices.append(tuple([b for i,b in enumerate(bi) if i != channel_axis]))
    else:
        label_block_indices = block_indices

    new_labeling = determine_merge_relabeling(label_block_indices, faces, box_ids,
                                              label_dist_th=label_dist_th)
    new_labeling_path = f'{output_dir}/new_labeling.npy'
    np.save(new_labeling_path, new_labeling)

    logger.info(f'Relabel {box_ids.shape} blocks from {new_labeling_path}')
    relabeled = da.map_blocks(
        lambda block: np.load(new_labeling_path)[block],
        segmentation_da,
        dtype=np.uint32,
        chunks=segmentation_da.chunks,
    )
    da.to_zarr(relabeled, f'{output_dir}/segmentation.zarr/remapped_block_labels', overwrite=True)
    merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids.astype(np.int32)])
    return relabeled, merged_boxes


def process_block(
    block_index,
    crop,
    image_container_path,
    image_subpath,
    image_shape,
    data_timeindex,
    data_channels,
    blocksize,
    blocksoverlap,
    labels_output_zarr,
    preprocessing_steps=[],
    model_type=None,
    diameter=30,
    eval_channels=None,
    do_3D=True,
    normalize=True,
    normalize_lowhigh=None,
    normalize_percentile=None,
    normalize_norm3D=True,
    normalize_sharpen_radius=0,
    normalize_smooth_radius=0,
    normalize_tile_norm_blocksize=0,
    normalize_tile_norm_smooth3D=1,
    normalize_invert=False,
    z_axis=0,
    channel_axis=None,
    anisotropy=None,
    min_size=15,
    resample=True,
    flow_threshold=0.4,
    cellprob_threshold=0,
    stitch_threshold=0,
    flow3D_smooth=1,
    use_gpu=False,
    gpu_device=None,
    gpu_batch_size=8,
    test_mode=False,
):
    """
    Preprocess and segment one block, of many, with eventual merger
    of all blocks in mind. The block is processed as follows:

    (1) Read block from disk, preprocess, and segment.
    (2) Remove overlaps.
    (3) Get bounding boxes for every segment.
    (4) Remap segment IDs to globally unique values.
    (5) Write segments to disk.
    (6) Get segmented block faces.

    A user may want to test this function on one block before running
    the distributed function. When test_mode=True, steps (5) and (6)
    are omitted and replaced with:

    (5) return remapped segments as a numpy array, boxes, and box_ids

    Parameters
    ----------
    block_index : tuple
        The (i, j, k, ...) index of the block in the overall block grid

    crop : tuple of slice objects
        The bounding box of the data to read from the input_zarr array

    image_container_path : string
        Path to image container.

    image_subpath : string
        Dataset path relative to image container.

    preprocessing_steps : list of tuples (default: the empty list)
        Optionally apply an arbitrary pipeline of preprocessing steps
        to the image block before running cellpose.

        Must be in the following format:
        [(f, {'arg1':val1, ...}), ...]
        That is, each tuple must contain only two elements, a function
        and a dictionary. The function must have the following signature:
        def F(image, ..., crop=None)
        That is, the first argument must be a numpy array, which will later
        be populated by the image data. The function must also take a keyword
        argument called crop, even if it is not used in the function itself.
        All other arguments to the function are passed using the dictionary.
        Here is an example:

        def F(image, sigma, crop=None):
            return gaussian_filter(image, sigma)
        def G(image, radius, crop=None):
            return median_filter(image, radius)
        preprocessing_steps = [(F, {'sigma':2.0}), (G, {'radius':4})]

    blocksize : iterable (list, tuple, np.ndarray)
        The number of voxels (the shape) of blocks without overlaps

    blocksoverlap : iterable (list, tuple, np.ndarray)
        The number of voxels added to the blocksize to provide context
        at the edges

    labels_output_zarr : zarr.core.Array
        A location where segments can be stored temporarily before
        merger is complete

    test_mode : bool (default: False)
        The primary use case of this function is to be called by
        distributed_eval (defined later in this same module). However
        you may want to call this function manually to test what
        happens to an individual block; this is a good idea before
        ramping up to process big data and also useful for debugging.

        When test_mode is False (default) this function stores
        the segments and returns objects needed for merging between
        blocks.

        When test_mode is True this function does not store the
        segments, and instead returns them to the caller as a numpy
        array. The boxes and box IDs are also returned. When test_mode
        is True, you can supply dummy values for many of the inputs,
        such as:

        block_index = (0, 0, 0)
        output_zarr=None

    Returns
    -------
    If test_mode == False (the default), three things are returned:
        faces : a list of numpy arrays - the faces of the block segments
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes

    If test_mode == True, three things are returned:
        segments : np.ndarray containing the segments with globally unique IDs
        boxes : a list of crops (tuples of slices), bounding boxes of segments
        box_ids : 1D numpy array, parallel to boxes, the segment IDs of the
                  boxes
    """
    logger.info((
        f'RUNNING BLOCK: {block_index}, '
        f'REGION: {crop}, '
        f'blocksize: {blocksize}, '
        f'blocksoverlap: {blocksoverlap}, '
        f'model_type: {model_type}, '
        f'do_3D: {do_3D}, '
        f'diameter: {diameter}, '
        f'eval_channels: {eval_channels}, '
        f'z_axis: {z_axis}, '
        f'channel_axis: {channel_axis}, '
        f'min_size: {min_size}, '
        f'resample: {resample}, '
        f'anisotropy: {anisotropy}, '
        f'flow_threshold: {flow_threshold}, '
        f'cellprob_threshold: {cellprob_threshold}, '
        f'stitch_threshold: {stitch_threshold}, '
        f'gpu_batch_size: {gpu_batch_size}, '
    ))
    segmentation = read_preprocess_and_segment(
        image_container_path, 
        image_subpath, 
        data_timeindex,
        data_channels,
        crop, 
        preprocessing_steps,
        model_type=model_type,
        diameter=diameter,
        eval_channels=eval_channels,
        z_axis=z_axis,
        channel_axis=channel_axis,
        do_3D=do_3D,
        normalize=normalize,
        normalize_lowhigh=normalize_lowhigh,
        normalize_percentile=normalize_percentile,
        normalize_norm3D=normalize_norm3D,
        normalize_sharpen_radius=normalize_sharpen_radius,
        normalize_smooth_radius=normalize_smooth_radius,
        normalize_tile_norm_blocksize=normalize_tile_norm_blocksize,
        normalize_tile_norm_smooth3D=normalize_tile_norm_smooth3D,
        normalize_invert=normalize_invert,
        min_size=min_size,
        resample=resample,
        anisotropy=anisotropy,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        flow3D_smooth=flow3D_smooth,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        gpu_batch_size=gpu_batch_size,
    )
    if (do_3D and len(image_shape) > 3 or
        not do_3D and len(image_shape) > 2):
        # labels are single channel so if the input was multichannel remove the channel coords
        labels_image_shape = [s for i, s in enumerate(image_shape) if i != channel_axis]
        labels_block_index = [b for i, b in enumerate(block_index) if i != channel_axis]
        labels_coords = [c for i, c in enumerate(crop) if i != channel_axis]
        labels_overlaps = [o for i, o in enumerate(blocksoverlap) if i != channel_axis]
        labels_blocksize = [s for i, s in enumerate(blocksize) if i != channel_axis]
    else:
        labels_image_shape = image_shape
        labels_block_index = block_index
        labels_coords = crop
        labels_overlaps = blocksoverlap
        labels_blocksize = blocksize

    logger.debug((
        f'adjusted labels image shape to {labels_image_shape} '
        f'labels block index to {labels_block_index} '
        f'labels block coords to {labels_coords} '
        f'labels block overlaps to {labels_overlaps} '
        f'labels block size to {labels_blocksize} '
    ))
    logger.debug(f'Segmented block shape before removing overlaps: {segmentation.shape}')
    segmentation, labels_coords = remove_overlaps(segmentation, labels_coords, labels_overlaps, labels_blocksize)
    boxes = bounding_boxes_in_global_coordinates(segmentation, labels_coords)
    nblocks = get_nblocks(labels_image_shape, labels_blocksize)
    segmentation, remap = global_segment_ids(segmentation, labels_block_index, nblocks)
    if remap[0] == 0:
        remap = remap[1:]

    labels_output_zarr[tuple(labels_coords)] = segmentation

    if test_mode:
        return segmentation, boxes, remap

    faces = block_faces(segmentation)
    return faces, boxes, remap


# ----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    image_container_path,
    image_subpath,
    data_timeindex,
    data_channels,
    crop,
    preprocessing_steps,
    # model_kwargs,
    model_type='cyto3',
    # eval_kwargs
    diameter=30,
    eval_channels=None,
    z_axis=0,
    channel_axis=None,
    do_3D=True,
    normalize=True,
    normalize_lowhigh=None,
    normalize_percentile=None,
    normalize_norm3D=True,
    normalize_sharpen_radius=0,
    normalize_smooth_radius=0,
    normalize_tile_norm_blocksize=0,
    normalize_tile_norm_smooth3D=1,
    normalize_invert=False,
    min_size=15,
    resample=True,
    anisotropy=None,
    flow_threshold=0.4,
    cellprob_threshold=0,
    stitch_threshold=0,
    flow3D_smooth=1,
    use_gpu=False,
    gpu_device=None,
    gpu_batch_size=8,
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""
    logger.info((
        f'Reading {crop} block from '
        f'{image_container_path}:{image_subpath} '
        f'timeindex {data_timeindex} '
        f'channels {data_channels} '
    ))
    image_block, block_attrs = read_utils.open(image_container_path, image_subpath,
                                               data_timeindex=data_timeindex,
                                               data_channels=data_channels,
                                               block_coords=crop)
    if anisotropy is None or anisotropy == 1:
        # try to compute it from block's attributes
        anisotropy = compute_block_anisotropy(block_attrs)

    start_time = time.time()

    for pp_step in preprocessing_steps:
        logger.debug(f'Apply preprocessing step: {pp_step}')
        image_block = pp_step[0](image_block, **pp_step[1])

    if use_gpu:
        available_gpus = torch.cuda.device_count()
        logger.info(f'Found {available_gpus} GPUs')
        if available_gpus > 1:
            # if multiple gpus are available try to find one that can be used
            segmentation_device, gpu = None, False
            for gpui in range(available_gpus):
                try:
                    logger.debug(f'Try GPU: {gpui}')
                    segmentation_device, gpu = cellpose.models.assign_device(gpu=use_gpu,
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
            segmentation_device, gpu = cellpose.models.assign_device(gpu=use_gpu,
                                                                     device=gpu_device)
    else:
        segmentation_device, gpu = cellpose.models.assign_device(gpu=use_gpu,
                                                                device=gpu_device)
    logger.info(f'Segmentation device for block {crop}: {segmentation_device}:{gpu}')
    model = cellpose.models.CellposeModel(gpu=gpu,
                                          model_type=model_type,
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
    if (do_3D and len(image_block.shape) == 3 or
        not do_3D and len(image_block.shape) == 2):
        # if 3D and the block has exactly 3 dimensions
        # or in the case of 2D segmentation the block has exactly 2 dimensions
        # reshape it to include a dimension for the channel
        new_block_shape = (1,) + image_block.shape
        logger.debug(f'Reshape block of {image_block.shape} to {new_block_shape}')
        image_block = image_block.reshape(new_block_shape)

    if normalize:
        logger.info(f'Normalize params: {normalize_params}')
        image_block = transforms.normalize_img(image_block, axis=channel_axis,
                                               **normalize_params)

    labels = model.eval(image_block,
                        channels=eval_channels,
                        diameter=diameter,
                        z_axis=z_axis,
                        channel_axis=channel_axis,
                        do_3D=do_3D,
                        min_size=min_size,
                        resample=resample,
                        anisotropy=anisotropy,
                        normalize=normalize_params,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        stitch_threshold=stitch_threshold,
                        batch_size=gpu_batch_size,
                        flow3D_smooth=flow3D_smooth,
                        )[0].astype(np.uint32)
    end_time = time.time()
    unique_labels = np.unique(labels)
    logger.info((
        f'Finished model eval for block: {crop} '
        f'found {len(unique_labels)} unique labels '
        f'in {end_time-start_time}s '
    ))
    return labels


def bounding_boxes_in_global_coordinates(segmentation, crop):
    """
    bounding boxes (tuples of slices) are super useful later
    best to compute them now while things are distributed
    """
    boxes = scipy.ndimage.find_objects(segmentation)
    boxes = [b for b in boxes if b is not None]

    def _translate(a, b):
        return slice(a.start+b.start, a.start+b.stop)

    for iii, box in enumerate(boxes):
        boxes[iii] = tuple(_translate(a, b) for a, b in zip(crop, box))
    return boxes


def global_segment_ids(segmentation, block_index, nblocks):
    """
    Pack the block index into the segment IDs so they are
    globally unique. Everything gets remapped to [1..N] later.
    A label is split into 5 digits on left and 5 digits on right.
    This creates limits: 42950 maximum number of blocks and
    99999 maximum number of segments per block
    """
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    logger.debug((
        f'Block {block_index} out of {nblocks} '
        f'- has {len(unique)} unique labels '
    ))
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [int(p+str(x).zfill(5)) for x in unique]
    if unique[0] == 0:
        remap[0] = 0  # 0 should just always be 0
    logger.debug(f'Remap: {remap}')
    segmentation = np.array(remap, dtype=np.uint32)[unique_inverse.reshape(segmentation.shape)]
    return segmentation, remap


def block_faces(segmentation):
    """Slice faces along every axis"""
    faces = []
    for iii in range(segmentation.ndim):
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(0, 1)
        faces.append(segmentation[tuple(a)])
        a = [slice(None),] * segmentation.ndim
        a[iii] = slice(-1, None)
        faces.append(segmentation[tuple(a)])
    return faces


def determine_merge_relabeling(block_indices, faces, labels,
                               label_dist_th=1.0):
    """Determine boundary segment mergers, remap all label IDs to merge
       and put all label IDs in range [1..N] for N global segments found"""
    faces = adjacent_faces(block_indices, faces)
    logger.debug(f'Determine relabeling for {labels.shape} of type {labels.dtype}')
    used_labels = labels.astype(int)
    label_range = int(np.max(used_labels) + 1)
    label_groups = block_face_adjacency_graph(faces, label_range,
                                              label_dist_th=label_dist_th)
    logger.debug((
        f'Build connected components for {label_groups.shape} label groups'
        f'{label_groups}'
    ))
    new_labeling = scipy.sparse.csgraph.connected_components(label_groups,
                                                             directed=False)[1]
    logger.debug(f'Initial {new_labeling.shape} connected labels:, {new_labeling}')
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique))[unique_inverse]
    logger.debug(f'Re-arranged {len(new_labeling)} connected labels:, {new_labeling}')
    return new_labeling


def adjacent_faces(block_indices, faces):
    """Find faces which touch and pair them together in new data structure"""
    face_pairs = []
    faces_index_lookup = {a: b for a, b in zip(block_indices, faces)}
    for block_index in block_indices:
        for ax in range(len(block_index)):
            neighbor_index = np.array(block_index)
            neighbor_index[ax] += 1
            neighbor_index = tuple(neighbor_index)
            try:
                a = faces_index_lookup[block_index][2*ax + 1]
                b = faces_index_lookup[neighbor_index][2*ax]
                face_pairs.append(np.concatenate((a, b), axis=ax))
            except KeyError:
                continue
    return face_pairs


def block_face_adjacency_graph(faces, labels_range, label_dist_th=1.0):
    """
    Shrink labels in face plane, then find which labels touch across the face boundary
    """
    logger.info(f'Create adjacency graph for {labels_range} labels')
    all_mappings = [np.empty((2, 0), dtype=np.uint32)]
    structure = scipy.ndimage.generate_binary_structure(3, 1)
    for face in faces:
        sl0 = tuple(slice(0, 1) if d == 2 else slice(None) for d in face.shape)
        sl1 = tuple(slice(1, 2) if d == 2 else slice(None) for d in face.shape)
        a = shrink_labels(face[sl0], label_dist_th)
        b = shrink_labels(face[sl1], label_dist_th)
        face = np.concatenate((a, b), axis=np.argmin(a.shape))
        mapped = di_ndmeasure._utils._label._across_block_label_grouping(
            face,
            structure
        )
        all_mappings.append(mapped)
    i, j = np.concatenate(all_mappings, axis=1)
    v = np.ones_like(i)
    csr_mat = scipy.sparse.coo_matrix((v, (i, j)),
                                      shape=(labels_range,labels_range)).tocsr()
    logger.debug(f'Labels mapping as csr matrix {csr_mat}')
    return csr_mat


def shrink_labels(plane, threshold):
    """
    Shrink labels in plane by some distance from their boundary
    """
    gradmag = np.linalg.norm(np.gradient(plane.squeeze()), axis=0)
    shrunk_labels = np.copy(plane.squeeze())
    shrunk_labels[gradmag > 0] = 0
    distances = scipy.ndimage.distance_transform_edt(shrunk_labels)
    shrunk_labels[distances <= threshold] = 0
    return shrunk_labels.reshape(plane.shape)


def merge_all_boxes(boxes, box_ids):
    """
    Merge all boxes that map to the same box_ids
    """
    merged_boxes = []
    boxes_array = np.array(boxes, dtype=object)
    for iii in np.unique(box_ids):
        merge_indices = np.argwhere(box_ids == iii).squeeze()
        if merge_indices.shape:
            merged_box = merge_boxes(boxes_array[merge_indices])
        else:
            merged_box = boxes_array[merge_indices]
        merged_boxes.append(merged_box)
    return merged_boxes


def merge_boxes(boxes):
    """Take union of two or more parallelpipeds"""
    box_union = boxes[0]
    for iii in range(1, len(boxes)):
        local_union = []
        for s1, s2 in zip(box_union, boxes[iii]):
            start = min(s1.start, s2.start)
            stop = max(s1.stop, s2.stop)
            local_union.append(slice(start, stop))
        box_union = tuple(local_union)
    return box_union
