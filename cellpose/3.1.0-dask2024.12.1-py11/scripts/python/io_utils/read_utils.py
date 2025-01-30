import logging
import nrrd
import numpy as np
import os
import zarr

from tifffile import TiffFile

from . import zarr_utils


logger = logging.getLogger(__name__)


def open(container_path, subpath, block_coords=None):
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
                               data_store_name=container_store)
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
