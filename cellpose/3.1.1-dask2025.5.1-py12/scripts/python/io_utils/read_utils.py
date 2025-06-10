import logging
import nrrd
import numpy as np
import os
import zarr

from tifffile import TiffFile

from . import zarr_utils


logger = logging.getLogger(__name__)


def get_voxel_spacing(attrs, default_value=None, as_zyx=True):
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


def open(container_path, subpath, subpath_pattern=None, block_coords=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        logger.info(f'Open nrrd {container_path} ({real_container_path})')
        return read_nrrd(container_path, block_coords=block_coords)
    elif container_ext == '.tif' or container_ext == '.tiff':
        logger.info(f'Open tiff {container_path} ({real_container_path})')
        im = read_tiff(container_path, block_coords=block_coords)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(container_path)
        return im, {}
    elif container_ext == '.n5' or container_ext == '.zarr':
        container_store = container_ext[1:]
        logger.info((
            f'Open {container_path} ({real_container_path}):{subpath} '
            f'using {container_store} container store '
        ))
        return zarr_utils.open(container_path, subpath,
                               block_coords=block_coords,
                               data_store_name=container_store,
                               data_subpath_pattern=subpath_pattern,)
    else:
        logger.warning(f'Cannot handle {container_path} ({real_container_path}): {subpath}')
        return None, {}


def read_tiff(input_path, block_coords=None):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        if block_coords is None:
            img = tif_array
        else:
            img = tif_array[block_coords]
        return img


def read_nrrd(input_path, block_coords=None):
    im, dict = nrrd.read(input_path)
    return im[block_coords] if block_coords is not None else im, dict
