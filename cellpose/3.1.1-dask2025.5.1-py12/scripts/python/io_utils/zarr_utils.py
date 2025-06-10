import logging
import os
import re
import zarr

from ome_zarr_models.v04.image import (ImageAttrs, Multiscale,
                                       Dataset)


logger = logging.getLogger(__name__)


def create_dataset(data_path, data_subpath, shape, chunks, dtype,
                   data=None, data_store_name=None,
                   container_attributes={},
                   **kwargs):
    try:
        data_store = _get_data_store(data_path, data_store_name)
        if data_subpath and data_subpath != '.':
            logger.info((
                f'Create dataset {data_path}:{data_subpath} '
                f'data store {data_store}'
            ))
            root_group = zarr.open_group(store=data_store, mode='a')
            if _is_ome_zarr(container_attributes):
                root_group.attributes = container_attributes
                ome_metadata = ImageAttrs(**container_attributes)
                datagroup_subpath, dataset_subpath = _get_dataset_subpath(data_subpath,
                                                                          ome_metadata.multiscales[0])
                if datagroup_subpath:
                    data_group = root_group.require_group(datagroup_subpath)
                else:
                    data_group = root_group
                data_group.attrs.update(container_attributes)
                # assume shape and chunks have the same length
                if len(shape) == len(ome_metadata.multiscales[0].axes):
                    dataset_shape = shape
                    dataset_chunks = chunks
                else:
                    missing_dims = len(ome_metadata.multiscales[0].axes)-len(shape)
                    dataset_shape = ((1,) * missing_dims + shape)
                    dataset_chunks = ((1,) * missing_dims + chunks)
            else:
                data_group = root_group
                dataset_subpath = data_subpath
                dataset_shape = shape
                dataset_chunks = chunks
            dataset = data_group.require_dataset(
                dataset_subpath,
                shape=dataset_shape,
                chunks=dataset_chunks,
                dtype=dtype,
                data=data)
            # set additional attributes
            dataset.attrs.update(**kwargs)
            return dataset
        else:
            logger.info((
                f'Create root array {data_path} '
                f'data store {data_store}'
            ))
            return zarr.open(store=data_store,
                             shape=shape,
                             chunks=chunks,
                             dtype=dtype,
                             mode='a')
    except Exception as e:
        logger.error(f'Error creating a dataset at {data_path}:{data_subpath}, {e}')
        raise e


def open(data_path, data_subpath, data_store_name=None,
         data_subpath_pattern=None,
         mode='r',
         block_coords=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        logger.debug(f'Open zarr container: {data_store}:{data_subpath} ({data_store_name})')
        data_container = zarr.open(store=data_store, mode=mode)
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            no_pattern = 'no'
            logger.info((
                f'Open OME ZARR {zarr_container_path}:{zarr_subpath} '
                f'with {data_subpath_pattern if data_subpath_pattern is not None else no_pattern} pattern '
            ))
            return _open_ome_zarr(data_container, data_container_attrs, zarr_subpath,
                                  data_subpath_pattern=data_subpath_pattern,
                                  block_coords=block_coords)
        else:
            a = (data_container[zarr_subpath] 
                if zarr_subpath and zarr_subpath != '.'
                else data_container)
            ba = a[block_coords] if block_coords is not None else a
            return ba, a.attrs.asdict()
    except Exception as e:
        logger.error(f'Error opening {data_path}:{data_subpath} {e}')
        raise e


def prepare_attrs(dataset_path, src_image_attributes:dict, **additional_attrs) -> dict:
    if not _is_ome_zarr(src_image_attributes):
        return {k: v for k, v in additional_attrs.items()}
    else:
        dataset_path_comps = [c for c in dataset_path.split('/') if c]
        dataset_scale_subpath = dataset_path_comps.pop()

        src_ome_metadata = ImageAttrs(**src_image_attributes)
        src_multiscale = src_ome_metadata.multiscales[0]

        if src_image_attributes.get('coordinateTransformations'):
            src_scales = src_image_attributes['coordinateTransformations'][0].scale
        else:
            src_scales = (1,) * len(src_multiscale.axes)
        if src_image_attributes.get('coordinateTransformations'):
            src_translations = src_image_attributes['coordinateTransformations'][1].translation
        else:
            src_translations = None

        dataset = Dataset.build(path=dataset_scale_subpath, scale=src_scales, translation=src_translations)
        ome_metadata = ImageAttrs(
            multiscales=[
                Multiscale(
                    axes=src_multiscale.axes,
                    datasets=(dataset,),
                )
            ],
        )
        ome_attrs = ome_metadata.dict()
        ome_attrs.update(additional_attrs)
        return ome_attrs


def _adjust_data_paths(data_path, data_subpath):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for the first subpath that containes .zattrs file
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if (os.path.exists(f'{container_path}/.zattrs') or
            os.path.exists(f'{container_path}/attributes.json')):
            break
        dataset_comps_index = dataset_comps_index + 1

    appended_container_path = '/'.join(dataset_comps[0:dataset_comps_index])
    container_path = f'{data_path}/{appended_container_path}'
    new_subpath = '/'.join(dataset_comps[dataset_comps_index:])

    return container_path, new_subpath


def _extract_numeric_comp(v):
    match = re.match(r'^(\D*)(\d+)$', v)
    if match:
        return int(match.groups()[1])
    else:
        raise ValueError(f'Invalid component: {v}')


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return zarr.DirectoryStore(data_path)


def _is_ome_zarr(data_container_attrs: dict | None) -> bool:
    if data_container_attrs is None:
        return False

    # test if multiscales attribute exists - if it does assume OME-ZARR
    multiscales = data_container_attrs.get('multiscales', [])
    return not (multiscales == [])


def _open_ome_zarr(data_container, data_container_attrs, data_subpath,
                   data_subpath_pattern=None,
                   block_coords=None):
    ome_zarr_metadata = ImageAttrs(**data_container_attrs)
    multiscale_metadata = ome_zarr_metadata.multiscales[0]
    dataset_metadata = None

    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    # lookup the dataset by path
    for ds in multiscale_metadata.datasets:
        current_ds_path_comps = [c for c in ds.path.split('/') if c]
        if (len(current_ds_path_comps) < len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the data_subpath arg
            dataset_metadata = ds
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            logger.debug((
                f'Found dataset: {dataset_metadata.path}, '
                f'remaining components: {dataset_comps}'
            ))
            break

    dataset_comps_pattern = (list(data_subpath_pattern) 
                             if data_subpath_pattern else [])
    ch = None
    timeindex = None
    dataset_axes = multiscale_metadata.axes

    for comp_index, comp in enumerate(dataset_comps_pattern):
        if (comp == 't' and
            any(a.type == 'time' for a in dataset_axes)):
            # if the time is present in the dataset subpath selector
            # and it is in the nd-array too - get the index to be processed
            logger.debug(f'Extract timeindex from {dataset_comps[comp_index]}')
            timeindex = _extract_numeric_comp(dataset_comps[comp_index])
        elif (comp == 'c' and
            any(a.type == 'channel' for a in dataset_axes)):
            # if the channel is present in the dataset subpath selector
            # and the nd-array is a multi-channel array get the channel to be processed
            logger.debug(f'Extract channel from {dataset_comps[comp_index]}')
            ch = _extract_numeric_comp(dataset_comps[comp_index])
        elif comp == 's' and dataset_metadata is None:
            # scale selector is in the path and dataset was not found
            # using the existing datasets paths
            logger.debug(f'Extract dataset index from {dataset_comps[comp_index]}')
            dataset_index = _extract_numeric_comp(dataset_comps[comp_index])
            dataset_metadata = multiscale_metadata.datasets[dataset_index]
        else:
            # this dataset component can be anything
            continue

    if dataset_metadata is None:
        dataset_metadata = multiscale_metadata.datasets[0]
        logger.info(f'No dataset was found so far - use the first one: {dataset_metadata.path}')

    dataset_path = dataset_metadata.path

    a = data_container[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selector(dataset_axes, timeindex, ch)(a)[block_coords]
    else:
        ba = _get_array_selector(dataset_axes, timeindex, ch)(a)
    data_container_attrs.update({
        'dataset_path': dataset_path,
        'axes': dataset_axes,
        'timeindex': timeindex,
        'channel': ch,
        'coordinateTransformations': dataset_metadata.coordinateTransformations
    })
    return ba, data_container_attrs


def _get_array_selector(axes, timeindex, ch):
    selector = []
    selection_exists = False
    for a in axes:
        if a.type == 'time':
            if timeindex is not None:
                selector.append(slice(timeindex,timeindex+1))
                selection_exists = True
            else:
                selector.append(slice(None, None))
        elif a.type == 'channel':
            if ch is not None:
                selector.append(slice(ch, ch+1))
                selection_exists = True
            else:
                selector.append(slice(None, None))
        else:
            selector.append(slice(None, None))

    return lambda a: a.get_basic_selection(tuple(selector)) if selection_exists else a


def _get_dataset_subpath(requested_subpath:str, multiscale: Multiscale, dataset_index=0) -> (str, str):
    dataset = multiscale.datasets[dataset_index]
    requested_subpath_comps = [c for c in requested_subpath.split('/') if c]

    if len(requested_subpath_comps) > 0:
        # drop the scale component
        requested_subpath_comps.pop()
    
    dataset_subpath = dataset.path

    return '/'.join(requested_subpath_comps), dataset_subpath
