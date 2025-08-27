import os
import zarr
import numpy as np
import re

from ome_zarr_models.v04.image import ImageAttrs
from tifffile import TiffFile


def get_voxel_spacing(attrs, default_value=[1., 1., 1.]):
    """
    Return voxel spacing as ZYX taking into consideration that:
    if it is an OME-ZARR it has the coords in TCZYX order
    and if it is an N5 or if the coordinates were passed in as defaults (using the default_value)
    then they be read as XYZ
    """

    if attrs.get('coordinateTransformations'):
        # this typically is the OME-ZARR case
        scale_metadata = list(filter(lambda t: t.get('type') == 'scale', attrs['coordinateTransformations']))
        if len(scale_metadata) > 0:
            dataset_scale = scale_metadata[0].get('scale', [1., 1., 1., 1., 1.])
            # get voxel spatial resolution as [dx, dy, dz]
            xyz_voxel_spacing = np.array(dataset_scale[-3:])[::-1]
        else:
            xyz_voxel_spacing = np.array(default_value) if default_value is not None else None
    elif (attrs.get('downsamplingFactors') is not None and
          attrs.get('pixelResolution') is not None):
        # this is the N5 case for a scale != S0
        xyz_voxel_spacing = (np.array(attrs['pixelResolution']) *
                             np.array(attrs['downsamplingFactors']))
    elif (attrs.get('pixelResolution')):
        # this is the N5 case for S0
        pixel_resolution = attrs['pixelResolution']
        if type(pixel_resolution) is list:
            xyz_voxel_spacing = np.array(pixel_resolution)
        else:
            raise ValueError((
                f'Unknown pixelResolution: {pixel_resolution} of type {type(pixel_resolution)} '
                f'found in {attrs} '
            ))
    else:
        xyz_voxel_spacing = np.array(default_value) if default_value is not None else None

    if xyz_voxel_spacing is not None:
        # flip the coordinates as zyx
        return xyz_voxel_spacing[::-1]
    else:
        return None


def open(container_path, subpath,
         data_timeindex=None, data_channels=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.tif' or container_ext == '.tiff':
        print(f'Open tiff {container_path} ({real_container_path})')
        return _open_tiff(real_container_path)
    elif (container_ext == '.n5'
          or os.path.exists(f'{real_container_path}/attributes.json')):
        print((
            f'Open N5 {container_path} ({real_container_path}) '
            f'subpath: {subpath} '
            f'timeindex: {data_timeindex} '
            f'channels: {data_channels} '
        ))
        return _open_zarr(real_container_path, subpath,
                          data_store_name='n5',
                          data_timeindex=data_timeindex,
                          data_channels=data_channels)
    elif container_ext == '.zarr':
        print((
            f'Open Zarr {container_path} ({real_container_path}) '
            f'subpath: {subpath} '
            f'timeindex: {data_timeindex} '
            f'channels: {data_channels} '
        ))
        return _open_zarr(real_container_path, subpath,
                          data_store_name='zarr',
                          data_timeindex=data_timeindex,
                          data_channels=data_channels)
    else:
        print(f'Cannot handle {container_path} ({real_container_path}) {subpath}')
        return None, {}


def _open_tiff(data_path):
    with TiffFile(data_path) as tif:
        tif_store = tif.aszarr()
        img = zarr.open(tif_store)
        return img, _get_tiff_attrs(img)


def _get_tiff_attrs(tif_array):
    dict = tif_array.attrs.asdict()
    dict.update({
        'dataType': tif_array.dtype,
        'dimensions': tif_array.shape,
    })
    return dict


def _open_zarr(data_path, data_subpath, data_store_name=None,
               data_timeindex=None, data_channels=None, 
               mode='r',
               block_coords=None):
    try:
        print(f'Opening {data_path}:{data_subpath}')
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath, data_store_name)
        data_store = _get_data_store(zarr_container_path, data_store_name)
        data_container = zarr.open(store=data_store, mode=mode)
        data_container_attrs = data_container.attrs.asdict()

        if _is_ome_zarr(data_container_attrs):
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
        raise e


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return zarr.NestedDirectoryStore(data_path)


def _adjust_data_paths(data_path, data_subpath, data_store_name):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    if data_store_name == 'n5' or data_path.endswith('.n5') or data_path.endswith('.N5'):
        # N5 container path is the same as the data_path
        # and the subpath is the dataset path
        return data_path, data_subpath

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


def _is_ome_zarr(data_container_attrs: dict) -> bool:
    if data_container_attrs is None:
        return False

    bioformats_layout = data_container_attrs.get("bioformats2raw.layout", None)
    multiscales = data_container_attrs.get('multiscales', [])
    return bioformats_layout == 3 or len(multiscales) > 0


def _find_ome_multiscales(data_container, data_subpath):
    print(f'Find OME multiscales group within {data_subpath}')
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
            print(f'Found multiscales at {group_subpath}: {dataset_item_attrs}')
            # found a group that has attributes which contain multiscales list
            return dataset_item, '/'.join(dataset_comps[dataset_comps_index:]), dataset_item_attrs

    data_container_attrs = data_container.attrs.asdict()
    if data_container_attrs.get('multiscales', []) == []:
        return None, None, {}
    else:
        # the container itself has multiscales attributes
        return data_container, '', data_container_attrs


def _open_ome_zarr(data_container, data_subpath,
                   data_timeindex=None, data_channels=None, block_coords=None):
    multiscales_group, dataset_subpath, multiscales_attrs  = _find_ome_multiscales(data_container, data_subpath)

    if multiscales_group is None:
        a = (data_container[data_subpath]
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()

    print((
        f'Open dataset {dataset_subpath}, timeindex: {data_timeindex}, '
        f'channels: {data_channels}, block_coords {block_coords} '
    ))

    dataset_comps = [c for c in dataset_subpath.split('/') if c]
    # ome_metadata = ImageAttrs.construct(**multiscales_attrs)
    multiscale_metadata = multiscales_attrs.get('multiscales', [])[0]
    # pprint.pprint(ome_metadata)
    dataset_metadata = None
    # lookup the dataset by path
    for ds in multiscale_metadata.get('datasets', []):
        ds_path = ds.get('path', '')
        current_ds_path_comps = [c for c in ds_path.split('/') if c]
        print((
            f'Compare current dataset path: {ds_path} ({current_ds_path_comps}) '
            f'with {dataset_subpath} ({dataset_comps}) '
        ))
        if (len(current_ds_path_comps) <= len(dataset_comps) and
            tuple(current_ds_path_comps) == tuple(dataset_comps[-len(current_ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            dataset_metadata = ds
            currrent_dataset_path = ds.get('path')
            # drop the matching suffix
            dataset_comps = dataset_comps[-len(current_ds_path_comps):]
            print((
                f'Found dataset: {currrent_dataset_path}, '
                f'remaining dataset components: {dataset_comps}'
            ))
            break

    if dataset_metadata is None:
        # could not find a dataset using the subpath 
        # look at the last subpath component and get the dataset index from there
        # e.g., if the subpath looks like:
        #       '/s<n>' => datasets[n] if n < len(datasets) otherwise datasets[0]
        dataset_index_comp = dataset_comps[-1]
        print(f'No dataset was found using {dataset_subpath} - try to use: {dataset_index_comp}')
        dataset_index = _extract_numeric_comp(dataset_index_comp)
        if dataset_index < len(multiscale_metadata.get('datasets', [])):
            dataset_metadata = multiscale_metadata['datasets'][dataset_index]
        else:
            dataset_metadata = multiscale_metadata['datasets'][0]

    dataset_axes = multiscale_metadata.get('axes')
    dataset_path = dataset_metadata.get('path')
    dataset_transformations = dataset_metadata.get('coordinateTransformations')
    print(f'Get array using array path: {dataset_path}:{data_timeindex}:{data_channels}')
    a = multiscales_group[dataset_path]
    # a is potentially a 5-dim array: [timepoint?, channel?, z, y, x]
    if block_coords is not None:
        ba = _get_array_selection(a, dataset_axes, data_timeindex, data_channels, block_coords)
    else:
        ba = _get_array_selection(a, dataset_axes, data_timeindex, data_channels, None)
    multiscales_attrs.update(a.attrs.asdict())
    multiscales_attrs.update({
        'dataset_path': dataset_path,
        'axes': dataset_axes,
        'dimensions': a.shape,
        'dataType': a.dtype,
        'blockSize': a.chunks,
        'timeindex': data_timeindex,
        'channels': data_channels,
        'coordinateTransformations': dataset_transformations,
    })
    return ba, multiscales_attrs


def _get_array_selection(arr, axes, timeindex: int | None,
                         ch:int | list[int] | None,
                         block_coords: tuple | None):
    ndim = arr.ndim

    if block_coords is None:
        coords_param = (slice(None,None),) * ndim
    elif len(block_coords) < ndim:
        coords_param = (slice(None,None),) * (ndim - len(block_coords)) + block_coords
    else:
        coords_param = block_coords

    selector = []
    selection_exists = False

    for ai, a in enumerate(axes):
        if a.get('type') == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(coords_param[ai])
        elif a.get('type') == 'channel':
            if ch is None or ch == []:
                selector.append(coords_param[ai])
            else:
                selector.append(ch)
                selection_exists = True
        else:
            selector.append(coords_param[ai])

        selection_exists = (selection_exists or
                            coords_param[ai].start is not None or
                            coords_param[ai].stop is not None)

    if selection_exists:
        try:
            # try to select the data using the selector
            return arr[tuple(selector)]
        except Exception  as e:
            print(f'Error selecting data with selector {selector} {e}')
            raise e
    else:
        return arr
