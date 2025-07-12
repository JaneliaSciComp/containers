import os
import zarr
import numpy as np

from tifffile import TiffFile

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
        data_store = _get_data_store(data_path, data_store_name)
        data_container = zarr.open(store=data_store, mode=mode)
        a = (data_container[data_subpath] 
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()
    except Exception as e:
        raise e


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return data_path


def get_voxel_spacing(attrs, default_value=None, as_zyx=False):
    print(f'Get voxel spacing attrs from {attrs}')
    if (attrs.get('downsamplingFactors')):
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
    elif default_value is not None:
        voxel_spacing = np.array(default_value)
    else:
        voxel_spacing = None
    if voxel_spacing is not None:
        # flip the coordinates if as_zyx is True
        return voxel_spacing[::-1] if as_zyx else voxel_spacing
    else:
        return None
