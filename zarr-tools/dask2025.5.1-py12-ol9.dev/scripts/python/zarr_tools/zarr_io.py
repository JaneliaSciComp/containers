import logging
import numcodecs as codecs
import os
import re
import zarr

from .ngff.ngff_utils import (get_dataset, get_datasets, get_multiscales, has_multiscales)
from typing import Tuple


logger = logging.getLogger(__name__)


def create_zarr_array(container_path:str,
                      data_subpath:str,
                      shape:Tuple[int],
                      chunks:Tuple[int],
                      dtype:str,
                      store_name:str|None=None,
                      compressor:str|None=None,
                      overwrite=False):

    real_container_path = os.path.realpath(container_path)
    if store_name == 'n5':
        store = zarr.N5Store(real_container_path)
    else:
        store = zarr.DirectoryStore(real_container_path, dimension_separator='/')

    codec = (None if compressor is None
             else codecs.get_codec(dict(id=compressor)))

    if data_subpath:
        root_group = zarr.open_group(store=store, mode='a')
        if overwrite:
            current_shape = shape
            zarray = root_group.create_dataset(
                data_subpath,
                shape = current_shape,
                chunks=chunks,
                dtype=dtype,
                overwrite=True,
                compressor=codec,
            )
        else:
            if data_subpath in root_group:
                # if the dataset already exists, get its shape
                current_shape = root_group[data_subpath].shape
                logger.info((
                    f'Dataset {container_path}:{data_subpath} '
                    f'already exists with shape {current_shape} '
                ))
            else:
                # this is a new dataset 
                current_shape = shape
            zarray = root_group.require_dataset(
                data_subpath,
                shape = current_shape, # use the current shape
                chunks=chunks,
                dtype=dtype,
                overwrite=True,
                compressor=codec,
            )
            _resize_zarr_array(zarray, shape)
            return zarray
    else:
        print('This is not supported yet')
        return None


def open_zarr(data_path:str, data_subpath:str, data_store_name:str|None=None, mode:str='r'):
    try:
        zarr_container, zarr_subpath = _get_data_store(data_path, data_subpath, data_store_name)

        print(f'Open zarr container: {zarr_container} ({zarr_subpath}), mode: {mode}')
        data_container = zarr.open(store=zarr_container, mode=mode)
        multiscales_group, dataset_subpath, multiscales_attrs  = _lookup_ome_multiscales(data_container, zarr_subpath)

        if multiscales_group is not None:
            print(f'Open OME ZARR {data_container}:{dataset_subpath}')
            return _open_ome_zarr(multiscales_group, dataset_subpath, multiscales_attrs)
        else:
            print(f'Open Simple ZARR {data_container}:{zarr_subpath}')
            return _open_simple_zarr(data_container, zarr_subpath)
    except Exception as e:
        print(f'Error opening {data_path}:{data_subpath} {e}')
        raise e


def _get_data_store(data_path, data_subpath, data_store_name):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    path_comps = os.path.splitext(data_path)
    ext = path_comps[1]
    if (ext is not None and ext.lower() == '.n5' or data_store_name == 'n5'):
        # N5 container path is the same as the data_path
        # and the subpath is the dataset path
        print(f'Create N5 store for {data_path}: {data_subpath}')
        return zarr.N5Store(data_path), data_subpath

    print(f'Create ZARR store for {data_path}: { data_subpath}')
    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for a valid container path - must contain
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if (os.path.exists(f'{container_path}/.zgroup') or
            os.path.exists(f'{container_path}/.zattrs') or
            os.path.exists(f'{container_path}/.zarray') or
            os.path.exists(f'{container_path}/attributes.json')):
            break
        dataset_comps_index = dataset_comps_index + 1

    appended_container_path = '/'.join(dataset_comps[0:dataset_comps_index])
    container_path = f'{data_path}/{appended_container_path}'
    new_subpath = '/'.join(dataset_comps[dataset_comps_index:])

    print(f'Found zarr container at {container_path}:{new_subpath}')
    return zarr.DirectoryStore(container_path, dimension_separator='/'), new_subpath


def _lookup_ome_multiscales(data_container, data_subpath):
    print(f'lookup OME multiscales group within {data_subpath}')
    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    dataset_comps_index = 1
    while dataset_comps_index < len(dataset_comps):
        container_item_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_item = data_container[container_item_subpath]
        container_item_attrs = container_item.attrs.asdict()

        if has_multiscales(container_item_attrs):
            print(f'Found multiscales at {container_item_subpath}: {container_item_attrs}')
            # found a group that has attributes which contain multiscales list
            return container_item, '/'.join(dataset_comps[dataset_comps_index:]), container_item_attrs
        else:
            dataset_comps_index = dataset_comps_index + 1

    # if no multiscales have found - look directly under root
    data_container_attrs = data_container.attrs.asdict()
    if has_multiscales(data_container_attrs):
        print(f'Found multiscales directly under root: {data_container_attrs}')
        # the container itself has multiscales attributes
        return data_container, data_subpath, data_container_attrs
    else:
        return None, None, {}


def _open_ome_zarr(multiscales_group, dataset_subpath, attrs):

    multiscale_metadata = get_multiscales(attrs)

    dataset_metadata = get_dataset(multiscale_metadata, dataset_subpath)

    if dataset_metadata is None:
        # could not find a dataset using the subpath 
        # look at the last subpath component and get the dataset index from there
        # e.g., if the subpath looks like:
        #       '/s<n>' => datasets[n] if n < len(datasets) otherwise datasets[0]
        dataset_comps = [c for c in dataset_subpath.split('/') if c]
        dataset_index_comp = dataset_comps[-1]
        print(f'No dataset was found using {dataset_subpath} - try to use: {dataset_index_comp}')
        datasets = get_datasets(multiscale_metadata)
        dataset_index = _extract_numeric_comp(dataset_index_comp)
        if dataset_index < len(datasets):
            dataset_metadata = datasets[dataset_index]
        elif len(datasets) > 0:
            dataset_metadata =datasets[0]
        else:
            raise ValueError(f'No datasets found in {attrs}')

    dataset_path = dataset_metadata.get('path')
    print(f'Get dataset using path: {dataset_path}')
    a = multiscales_group[dataset_path] if dataset_path else multiscales_group
    _set_array_attrs(attrs, dataset_path, a.shape, a.dtype, a.chunks)

    return multiscales_group, attrs, dataset_path


def _extract_numeric_comp(v):
    match = re.match(r'^(\D*)(\d+)$', v)
    if match:
        return int(match.groups()[1])
    else:
        raise ValueError(f'Invalid component: {v}')


def _open_simple_zarr(data_container, data_subpath):
    a = (data_container[data_subpath] 
        if data_subpath and data_subpath != '.'
        else data_container)
    dataset_comps = [c for c in data_subpath.split('/') if c]
    parent_group_subpath = '/'.join(dataset_comps[:-1])
    if parent_group_subpath == '':
        parent_group = data_container
    else:
        parent_group = data_container[parent_group_subpath]

    attrs = parent_group.attrs.asdict()
    _set_array_attrs(attrs, data_subpath, a.shape, a.dtype, a.chunks)
    return parent_group, attrs, (dataset_comps[-1] if len(dataset_comps) > 0 else '')


def _set_array_attrs(attrs, subpath, shape, dtype, chunks):
    """
    Add useful datasets attributes from the array attributes:
    shape, ndims, data_type, chunksize
    """
    attrs.update({
        'dataset_path': subpath,
        'dataset_shape': shape,
        'dataset_dims': len(shape),
        'dataset_dtype': dtype.name,
        'dataset_blocksize': chunks,
    })
    return attrs


def _resize_zarr_array(zarray, new_shape):
    """
    Resize the array to fit the new shape
    """
    if zarray.shape != new_shape:
        zarray.resize(new_shape)
