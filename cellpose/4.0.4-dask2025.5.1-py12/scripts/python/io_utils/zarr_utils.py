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
         data_timeindex=None, data_channels=None, 
         mode='r',
         block_coords=None):
    try:
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        logger.debug(f'Open zarr container: {data_store}:{data_subpath} ({data_store_name})')
        data_container = zarr.open(store=data_store, mode=mode)
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
            logger.info((
                f'Open OME ZARR {zarr_container_path}:{zarr_subpath} '
                f'timeindex: {data_timeindex} '
                f'channels: {data_channels} '
            ))
            return _open_ome_zarr(data_container, zarr_subpath,
                                  data_timeindex=data_timeindex,
                                  data_channels=data_channels,
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


def prepare_attrs(dataset_path, # src_image_attributes:dict,
                  axes:list | None = None,
                  coordinateTransformations:list | None = None,
                  **additional_attrs) -> dict:
    if (coordinateTransformations is None or coordinateTransformations == []
        or axes is None):
        # coordinateTransformation is None or [] or no axes were provided
        return {k: v for k, v in additional_attrs.items()}
    else:
        dataset_path_comps = [c for c in dataset_path.split('/') if c]
        # take the last component of the dataset path to be the scale path
        dataset_scale_subpath = dataset_path_comps.pop()

        scales, translations = (1,) * len(axes), None
        for t in coordinateTransformations:
            if t.type == 'scale':
                scales = t.scale
            elif t.type == 'translation':
                translations = t.translation

        dataset = Dataset.build(path=dataset_scale_subpath, scale=scales, translation=translations)
        ome_metadata = ImageAttrs(
            multiscales=[
                Multiscale(
                    axes=axes,
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
    bioformats_layout = data_container_attrs.get("bioformats2raw.layout", None)
    multiscales = data_container_attrs.get('multiscales', [])
    return bioformats_layout == 3 or not (multiscales == [])


def _find_ome_multiscales(data_container, data_subpath):
    logger.info(f'Find OME multiscales group within {data_subpath}')
    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    dataset_comps_index = 0
    while dataset_comps_index < len(dataset_comps):
        group_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        dataset_item = data_container[group_subpath]
        dataset_item_attrs = dataset_item.attrs.asdict()
        if dataset_item_attrs.get('multiscales', []) == []:
            dataset_comps_index = dataset_comps_index + 1
        else:
            logger.debug(f'Found multiscales at {group_subpath}: {dataset_item_attrs}')
            # found a group that has attributes which contain multiscales list
            return dataset_item, '/'.join(dataset_comps[dataset_comps_index:]), dataset_item_attrs

    return None, None, {}


def _open_ome_zarr(data_container, data_subpath,
                   data_timeindex=None, data_channels=None, block_coords=None):
    multiscales_group, dataset_subpath, multiscales_attrs  = _find_ome_multiscales(data_container, data_subpath)

    if multiscales_group is None:
        a = (data_container[data_subpath]
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()

    logger.info((
        f'Open dataset {dataset_subpath}, timeindex: {data_timeindex}, '
        f'channels: {data_channels}, block_coords {block_coords} '
    ))

    dataset_comps = [c for c in dataset_subpath.split('/') if c]
    ome_metadata = ImageAttrs(**multiscales_attrs)
    multiscale_metadata = ome_metadata.multiscales[0]
    dataset_metadata = None
    # lookup the dataset by path
    for ds in multiscale_metadata.datasets:
        current_ds_path_comps = [c for c in ds.path.split('/') if c]
        if (len(current_ds_path_comps) < len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            dataset_metadata = ds
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            logger.info((
                f'Found dataset: {dataset_metadata.path}, '
                f'remaining dataset components: {dataset_comps}'
            ))
            break

    if dataset_metadata is None:
        # could not find a dataset using the subpath 
        # look at the last subpath component and get the dataset index from there
        # e.g., if the subpath looks like:
        #       '/s<n>' => datasets[n] if n < len(datasets) otherwise datasets[0]
        dataset_index_comp = dataset_comps[-1]
        logger.info(f'No dataset was found using {dataset_subpath} - try to use: {dataset_index_comp}')
        dataset_index = _extract_numeric_comp(dataset_index_comp)
        if dataset_index < len(multiscale_metadata.datasets):
            dataset_metadata = multiscale_metadata.datasets[dataset_index]
        else:
            dataset_metadata = multiscale_metadata.datasets[0]

    dataset_axes = multiscale_metadata.axes
    dataset_path = dataset_metadata.path
    logger.debug(f'Get array using array path: {dataset_path}:{data_timeindex}:{data_channels}')
    a = multiscales_group[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selector(dataset_axes, data_timeindex, data_channels)(a)[block_coords]
    else:
        ba = _get_array_selector(dataset_axes, data_timeindex, data_channels)(a)
    multiscales_attrs.update({
        'dataset_path': dataset_path,
        'axes': dataset_axes,
        'timeindex': data_timeindex,
        'channels': data_channels,
        'coordinateTransformations': dataset_metadata.coordinateTransformations
    })
    return ba, multiscales_attrs


def _get_array_selector(axes, timeindex: int|None, ch:int | list[int] | None):
    selector = []
    selection_exists = False
    for a in axes:
        if a.type == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(slice(None, None))
        elif a.type == 'channel':
            if ch is None or ch == []:
                selector.append(slice(None, None))
            else:
                selector.append(ch)
                selection_exists = True
        else:
            selector.append(slice(None, None))
    return lambda a: a[tuple(selector)] if selection_exists else a


def _get_dataset_subpath(requested_subpath:str, multiscale: Multiscale, dataset_index=0) -> (str, str):
    dataset = multiscale.datasets[dataset_index]
    requested_subpath_comps = [c for c in requested_subpath.split('/') if c]

    if len(requested_subpath_comps) > 0:
        # drop the scale component
        requested_subpath_comps.pop()
    
    dataset_subpath = dataset.path

    return '/'.join(requested_subpath_comps), dataset_subpath
