import os
import zarr
import numpy as np
import re

from ome_zarr_models.v04.image import ImageAttrs
from tifffile import TiffFile


def get_voxel_spacing(attrs, default_value=None, as_zyx=False):
    print(f'Get voxel spacing attrs from {attrs}')
    if attrs.get('coordinateTransformations'):
        scale_metadata = list(filter(lambda t: t.type == 'scale', attrs['coordinateTransformations']))
        if len(scale_metadata) > 0:
            # return voxel spacing as [dx, dy, dz]
            voxel_spacing = np.array(scale_metadata[0].scale[2:][::-1])
        else:
            voxel_spacing = np.array(default_value) if default_value is not None else None
    elif (attrs.get('downsamplingFactors')):
        voxel_spacing = (np.array(attrs['pixelResolution']) * 
                         np.array(attrs['downsamplingFactors']))
    elif (attrs.get('pixelResolution')):
        pixel_resolution = attrs['pixelResolution']
        if type(pixel_resolution) is list:
            voxel_spacing = np.array(pixel_resolution)
        elif type(pixel_resolution) is dict:
            voxel_spacing = np.array(pixel_resolution['dimensions'])
        else:
            raise ValueError(f'Unknown pixelResolution: {pixel_resolution} of type {type(pixel_resolution)}')
    else:
        voxel_spacing = np.array(default_value) if default_value is not None else None

    if voxel_spacing is not None:
        # flip the coordinates if as_zyx is True
        return voxel_spacing[::-1] if as_zyx else voxel_spacing
    else:
        return None


def open(container_path, subpath):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.tif' or container_ext == '.tiff':
        print(f'Open tiff {container_path} ({real_container_path})')
        return _open_tiff(real_container_path)
    elif (container_ext == '.n5'
          or os.path.exists(f'{real_container_path}/attributes.json')):
        print(f'Open N5 {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath, data_store_name='n5')
    elif container_ext == '.zarr':
        print(f'Open Zarr {container_path} ({real_container_path})')
        return _open_zarr(real_container_path, subpath, data_store_name='zarr')
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
         mode='r',
         block_coords=None):
    try:
        print(f'Opening {data_path}:{data_subpath}')
        zarr_container_path, zarr_subpath = _adjust_data_paths(data_path, data_subpath)
        data_store = _get_data_store(zarr_container_path, data_store_name)
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
        raise e


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return zarr.DirectoryStore(data_path)


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
    # a is a 5-dim array: [timepoint, channel, z, y, x]
    ba = (a[timeindex, ch][block_coords] if block_coords is not None
                                         else da.from_zarr(a)[timeindex, ch])
    data_container_attrs.update({
        'coordinateTransformations': dataset_metadata.coordinateTransformations
    })
    return ba, data_container_attrs

