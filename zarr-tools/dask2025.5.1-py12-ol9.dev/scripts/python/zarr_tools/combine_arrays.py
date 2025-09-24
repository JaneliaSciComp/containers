import logging
import traceback
import zarr

from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, as_completed
from toolz import partition_all
from typing import List, Tuple


logger = logging.getLogger(__name__)


def combine_arrays(input_zarrays: List[Tuple[zarr.Array, int, int]],
                   output_zarray:zarr.Array,
                   client: Client,
                   partition_size=100000):
    """
    Combine arrays
    """
    block_slices = slices_from_chunks(normalize_chunks(output_zarray.chunks, shape=output_zarray.shape))
    partitioned_block_slices = tuple(partition_all(partition_size, block_slices))
    logger.info(f'Partition {len(block_slices)} into {len(partitioned_block_slices)} partitions of up to {partition_size} blocks')

    for idx, part in enumerate(partitioned_block_slices):
        logger.info(f'Process partition {idx}')
        input_blocks = client.map(_read_input_blocks, part, source_arrays=input_zarrays)
        res = client.map(_write_blocks, input_blocks, output=output_zarray)

        for f, r in as_completed(res, with_results=True):
            if f.cancelled():
                exc = f.exception()
                logger.exception(f'Block processing exception: {exc}')
                res = False
            else:
                logger.debug(f'Finished writing blocks {r}')

        logger.info(f'Finished partition {idx}')


def _read_input_blocks(coords, source_arrays=[]):
    return [(coords, ch, tp, arr[coords[-3:]]) for (arr, ch, tp) in source_arrays]


def _write_blocks(blocks, output=[]):
    written_blocks = []
    for (coords, ch, tp, block) in blocks:
        if tp is not None:
            block_coords = (tp, ch) + coords[-3:]
        else:
            block_coords = (ch,) + coords[-3:]
        # write the block
        output[block_coords] = block
        written_blocks.append(block_coords)

    return written_blocks
