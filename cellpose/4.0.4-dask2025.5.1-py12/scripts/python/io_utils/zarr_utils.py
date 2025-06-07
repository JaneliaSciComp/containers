import logging
import os
import re
import zarr

from ome_zarr_models.v04.image import ImageAttrs


logger = logging.getLogger(__name__)


def create_dataset(data_path, data_subpath, shape, chunks, dtype,
                   data=None, data_store_name=None,
                   **kwargs):
    try:
        data_store = _get_data_store(data_path, data_store_name)
        if data_subpath and data_subpath != '.':
            logger.info((
                f'Create dataset {data_path}:{data_subpath} '
                f'data store {data_store}'
            ))
            root_group = zarr.open_group(store=data_store, mode='a')
            dataset = root_group.require_dataset(
                data_subpath,
                shape=shape,
                chunks=chunks,
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
         mode='r',
         block_coords=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        logger.debug(f'Open zarr container: {data_store}:{data_subpath} ({data_store_name})')
        data_container = zarr.open(store=data_store, mode=mode)
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            return _open_ome_zarr(data_container, data_container_attrs, zarr_subpath,
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


def _is_ome_zarr(data_container_attrs: dict) -> bool:
    # test if multiscales attribute exists - if it does assume OME-ZARR
    multiscales = data_container_attrs.get('multiscales', [])
    return not (multiscales == [])


def _open_ome_zarr(data_container, data_container_attrs, data_subpath, block_coords=None):
    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    if len(dataset_comps) > 0:
        scale = _extract_numeric_comp(dataset_comps[-1])
    else:
        scale = 0
    if len(dataset_comps) > 1:
        ch = _extract_numeric_comp(dataset_comps[-2])
    else:
        ch = 0
    ome_zarr_metadata = ImageAttrs(**data_container_attrs)
    multiscale_metadata = ome_zarr_metadata.multiscales[0]
    dataset_metadata = multiscale_metadata.datasets[scale]
    dataset_path = dataset_metadata.path
    # for now always use 0 for the timepoint and assume it is coded in the dataset path
    timeindex = 0
    a = data_container[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selector(multiscale_metadata, timeindex, ch)(a)[block_coords]
    else:
        ba = _get_array_selector(multiscale_metadata, timeindex, ch)(a)
    data_container_attrs.update({
        'coordinateTransformations': dataset_metadata.coordinateTransformations
    })
    return ba, data_container_attrs


def _get_array_selector(metadata, timepoint, ch):
    axes = metadata.axes
    has_time_dimension = any(a.type == 'time' for a in axes)
    has_channel_dimension = any(a.type == 'time' for a in axes)

    def _selector(a):
        if has_time_dimension:
            sa = a[timepoint]
        else:
            sa = a
        if has_channel_dimension:
            return sa[ch]
        else:
            return sa

    return _selector
