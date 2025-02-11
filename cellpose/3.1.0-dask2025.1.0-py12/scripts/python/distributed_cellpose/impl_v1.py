"""
This is code contributed by Greg Fleishman to run Cellpose on a Dask cluster.
"""
import cellpose.io
import cellpose.models
import dask.array as da
import dask_image.ndmeasure as di_ndmeasure
import logging
import numpy as np
import scipy
import time

import io_utils.read_utils as read_utils
import io_utils.zarr_utils as zarr_utils


logger = logging.getLogger(__name__)


def distributed_eval(
        image_container_path,
        image_subpath,
        image_shape,
        model_type,
        diameter,
        blocksize,
        output_dir,
        dask_client,
        blocksoverlap=(),
        mask=None,
        preprocessing_steps=[],
        use_torch=False,
        use_gpu=False,
        gpu_device=None,
        eval_channels=None,
        do_3D=True,
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
        eval_model_with_size=True,
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

    segmentation_zarr_container = f'{output_dir}/segmentation.zarr'
    labels_zarr = zarr_utils.create_dataset(
        segmentation_zarr_container,
        'block_labels',
        image_shape,
        blocksize,
        np.uint32,
        data_store_name='zarr',
    )

    logger.info(f'Start segmenting: {len(block_indices)} {blocksize} blocks with overlap {blocksoverlap}')
    futures = dask_client.map(
        process_block,
        block_indices,
        block_crops,
        image_container_path=image_container_path,
        image_subpath=image_subpath,
        image_shape=image_shape,
        blocksize=blocksize,
        blocksoverlap=blocksoverlap_arr,
        labels_output_zarr=labels_zarr,
        preprocessing_steps=preprocessing_steps,
        model_type=model_type,
        use_torch=use_torch,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        diameter=diameter,
        eval_channels=eval_channels,
        do_3D=do_3D,
        z_axis=z_axis,
        channel_axis=channel_axis,
        anisotropy=anisotropy,
        min_size=min_size,
        resample=resample,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        gpu_batch_size=gpu_batch_size,
        eval_model_with_size=eval_model_with_size,
        test_mode=test_mode,
    )

    results = dask_client.gather(futures)
    faces, boxes_, box_ids_ = list(zip(*results))
    segmentation_da = da.from_zarr(labels_zarr)

    if test_mode:
        # return the labeled blocks before merging the labels
        logger.info('Return labeled blocks')
        return segmentation_da, []

    boxes = [box for sublist in boxes_ for box in sublist]
    box_ids = np.concatenate(box_ids_)
    new_labeling = determine_merge_relabeling(block_indices, faces, box_ids,
                                              label_dist_th=label_dist_th)
    new_labeling_path = f'{output_dir}/new_labeling.npy'
    np.save(new_labeling_path, new_labeling)

    logger.info(f'Relabel blocks from {new_labeling_path}')
    relabeled = da.map_blocks(
        lambda block: np.load(new_labeling_path)[block],
        segmentation_da,
        dtype=np.uint32,
        chunks=segmentation_da.chunks,
    )
    da.to_zarr(relabeled, f'{output_dir}/segmentation.zarr/remapped_block_labels', overwrite=True)
    merged_boxes = merge_all_boxes(boxes, new_labeling[box_ids])
    return relabeled, merged_boxes


def process_block(
    block_index,
    crop,
    image_container_path,
    image_subpath,
    image_shape,
    blocksize,
    blocksoverlap,
    labels_output_zarr,
    preprocessing_steps=[],
    model_type='cyto3',
    use_torch=False,
    use_gpu=False,
    gpu_device=None,
    diameter=30,
    eval_channels=None,
    do_3D=True,
    z_axis=0,
    channel_axis=None,
    anisotropy=None,
    min_size=15,
    resample=True,
    flow_threshold=0.4,
    cellprob_threshold=0,
    stitch_threshold=0,
    gpu_batch_size=8,
    eval_model_with_size=True,
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
    logger.info(f'RUNNING BLOCK: {block_index},\tREGION: {crop}')
    segmentation = read_preprocess_and_segment(
        image_container_path, 
        image_subpath, 
        crop, 
        preprocessing_steps,
        model_type=model_type,
        use_torch=use_torch,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
        diameter=diameter,
        eval_channels=eval_channels,
        z_axis=z_axis,
        channel_axis=channel_axis,
        do_3D=do_3D,
        min_size=min_size,
        resample=resample,
        anisotropy=anisotropy,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        gpu_batch_size=gpu_batch_size,
        eval_model_with_size=eval_model_with_size,
    )
    segmentation, crop = remove_overlaps(segmentation, crop, blocksoverlap, blocksize)
    boxes = bounding_boxes_in_global_coordinates(segmentation, crop)
    nblocks = get_nblocks(image_shape, blocksize)
    segmentation, remap = global_segment_ids(segmentation, block_index, nblocks)
    if remap[0] == 0:
        remap = remap[1:]

    labels_output_zarr[tuple(crop)] = segmentation

    if test_mode:
        return segmentation, boxes, remap

    faces = block_faces(segmentation)
    return faces, boxes, remap


# ----------------------- component functions ---------------------------------#
def read_preprocess_and_segment(
    image_container_path,
    image_subpath,
    crop,
    preprocessing_steps,
    # model_kwargs,
    model_type='cyto3',
    use_torch=False,
    use_gpu=False,
    gpu_device=None,
    # eval_kwargs
    diameter=30,
    eval_channels=None,
    z_axis=0,
    channel_axis=None,
    do_3D=True,
    min_size=15,
    resample=True,
    anisotropy=None,
    flow_threshold=0.4,
    cellprob_threshold=0,
    stitch_threshold=0,
    gpu_batch_size=8,
    eval_model_with_size=True,
):
    """Read block from zarr array, run all preprocessing steps, run cellpose"""
    logger.info((
        f'Reading {crop} block from '
        f'{image_container_path}:{image_subpath} '
    ))
    image_block, _ = read_utils.open(image_container_path, image_subpath,
                                     block_coords=crop)

    start_time = time.time()

    for pp_step in preprocessing_steps:
        logger.debug(f'Apply preprocessing step: {pp_step}')
        image_block = pp_step[0](image_block, **pp_step[1])

    segmentation_device, gpu = cellpose.models.assign_device(use_torch=use_torch,
                                                             gpu=use_gpu,
                                                             device=gpu_device)
    if eval_model_with_size:
        model = cellpose.models.Cellpose(gpu=gpu,
                                         model_type=model_type,
                                         device=segmentation_device)
    else:
        model = cellpose.models.CellposeModel(gpu=gpu,
                                              model_type=model_type,
                                              device=segmentation_device)
    labels = model.eval(image_block, 
                        channels=eval_channels,
                        diameter=diameter,
                        z_axis=z_axis,
                        channel_axis=channel_axis,
                        do_3D=do_3D,
                        min_size=min_size,
                        resample=resample,
                        anisotropy=anisotropy,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        stitch_threshold=stitch_threshold,
                        batch_size=gpu_batch_size,
                        )[0].astype(np.uint32)

    end_time = time.time()

    logger.info((
        f'Finished model eval for block: {crop} '
        f'in {end_time-start_time}s '
    ))
    return labels


def get_block_crops(shape, blocksize, overlaps, mask):
    """
    Given a voxel grid shape, blocksize, and overlap size, construct
       tuples of slices for every block; optionally only include blocks
       that contain foreground in the mask. Returns parallel lists,
       the block indices and the slice tuples.
    """
    blocksize = np.array(blocksize, dtype=int)
    blockoverlaps = np.array(overlaps, dtype=int)

    if mask is not None:
        ratio = np.array(mask.shape) / shape
        mask_blocksize = np.round(ratio * blocksize).astype(int)

    indices, crops = [], []
    nblocks = get_nblocks(shape, blocksize)
    for index in np.ndindex(*nblocks):
        start = blocksize * index - blockoverlaps
        stop = start + blocksize + 2 * blockoverlaps
        start = np.maximum(0, start)
        stop = np.minimum(shape, stop)
        crop = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if mask is not None:
            start = mask_blocksize * index
            stop = start + mask_blocksize
            stop = np.minimum(mask.shape, stop)
            mask_crop = tuple(slice(x, y) for x, y in zip(start, stop))
            if not np.any(mask[mask_crop]):
                foreground = False
        if foreground:
            indices.append(index)
            crops.append(crop)
    return indices, crops


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def remove_overlaps(array, crop, overlaps, blocksize):
    """
    Overlaps are only there to provide context for boundary voxels
    and can be removed after segmentation is complete
    reslice array to remove the overlaps
    """
    logger.debug((
        f'Remove overlaps: {overlaps} '
        f'crop: {crop} '
        f'blocksize is {blocksize} '
        f'block shape: {array.shape} '
    ))
    crop_trimmed = list(crop)
    for axis in range(array.ndim):
        if crop[axis].start != 0:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(overlaps[axis], None)
            loverlap_index = tuple(slc)
            logger.debug((
                f'Remove left overlap on axis {axis}: {loverlap_index} ({type(loverlap_index)}) '
                f'from labeled block of shape: {array.shape} '
            ))
            array = array[loverlap_index]
            a, b = crop[axis].start, crop[axis].stop
            crop_trimmed[axis] = slice(a + overlaps[axis], b)
        if array.shape[axis] > blocksize[axis]:
            slc = [slice(None),]*array.ndim
            slc[axis] = slice(None, blocksize[axis])
            roverlap_index = tuple(slc)
            logger.debug((
                f'Remove right overlap on axis {axis}: {roverlap_index} ({type(roverlap_index)}) '
                f'from labeled block of shape: {array.shape} '
            ))
            array = array[roverlap_index]
            a = crop_trimmed[axis].start
            crop_trimmed[axis] = slice(a, a + blocksize[axis])
    return array, crop_trimmed


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


def get_nblocks(shape, blocksize):
    """Given a shape and blocksize determine the number of blocks per axis"""
    return np.ceil(np.array(shape) / blocksize).astype(int)


def global_segment_ids(segmentation, block_index, nblocks):
    """
    Pack the block index into the segment IDs so they are
    globally unique. Everything gets remapped to [1..N] later.
    A uint32 is split into 5 digits on left and 5 digits on right.
    This creates limits: 42950 maximum number of blocks and
    99999 maximum number of segments per block
    """
    logger.debug(f'Get global segment ids for block {block_index} - start at: {nblocks}')
    unique, unique_inverse = np.unique(segmentation, return_inverse=True)
    p = str(np.ravel_multi_index(block_index, nblocks))
    remap = [np.uint32(p+str(x).zfill(5)) for x in unique]
    if unique[0] == 0:
        remap[0] = np.uint32(0)  # 0 should just always be 0
    segmentation = np.array(remap)[unique_inverse.reshape(segmentation.shape)]
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


def determine_merge_relabeling(block_indices, faces, used_labels,
                               label_dist_th=1.0):
    """Determine boundary segment mergers, remap all label IDs to merge
       and put all label IDs in range [1..N] for N global segments found"""
    faces = adjacent_faces(block_indices, faces)
    label_range = np.max(used_labels)
    label_groups = block_face_adjacency_graph(faces, label_range,
                                              label_dist_th=label_dist_th)
    new_labeling = scipy.sparse.csgraph.connected_components(
        label_groups, directed=False)[1]
    logger.debug(f'Connected labels: {new_labeling}')
    # XXX: new_labeling is returned as int32. Loses half range. Potentially a problem.
    unused_labels = np.ones(label_range + 1, dtype=bool)
    unused_labels[used_labels] = 0
    new_labeling[unused_labels] = 0
    unique, unique_inverse = np.unique(new_labeling, return_inverse=True)
    new_labeling = np.arange(len(unique), dtype=np.uint32)[unique_inverse]
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


def block_face_adjacency_graph(faces, nlabels, label_dist_th=1.0):
    """
    Shrink labels in face plane, then find which labels touch across the face boundary
    """
    all_mappings = []
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
                                      shape=(nlabels+1,
                                      nlabels+1)).tocsr()
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
