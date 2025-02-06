import logging
import numpy as np
import zarr


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
        data_store = _get_data_store(data_path, data_store_name)
        logger.debug(f'Open zarr container: {data_store}:{data_subpath} ({data_store_name})')
        data_container = zarr.open(store=data_store, mode=mode)
        a = (data_container[data_subpath] 
             if data_subpath and data_subpath != '.'
             else data_container)
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()
    except Exception as e:
        logger.error(f'Error opening {data_path}:{data_subpath} {e}')
        raise e


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return data_path
