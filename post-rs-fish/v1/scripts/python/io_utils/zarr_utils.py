import zarr
import numpy as np


def open(data_path, data_subpath, data_store_name=None,
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
