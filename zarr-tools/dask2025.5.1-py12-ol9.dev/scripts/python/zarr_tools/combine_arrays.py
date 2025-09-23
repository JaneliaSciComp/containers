import functools
import numpy as np
import re
import zarr

from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client
from typing import List, Tuple




def combine_arrays(input_zarrays: List[Tuple[zarr.Array, int, int]],
                   output_zarray:zarr.Array,
                   client: Client):
    """
    Combine arrays
    """
    block_slices = slices_from_chunks(normalize_chunks(output_zarray.chunks, shape=output_zarray.shape))
    input_blocks = client.map(_read_input_blocks, block_slices, source_arrays=input_zarrays)
    res = client.map(_write_blocks, input_blocks, output=output_zarray)

    client.gather(res)
    print('!!!!! DONE')


def _read_input_blocks(coords, source_arrays=[]):
    print('!!!!! COORDS: ', coords)
    return [(coords, ch, tp, arr[coords[-3:]]) for (arr, ch, tp) in source_arrays]


def _write_blocks(blocks, output=[]):
    for (coords, ch, tp, block) in blocks:
        if tp is not None:
            block_coords = (tp, ch) + coords[-3:]
        else:
            block_coords = (ch,) + coords[-3:]
        # write the block
        output[block_coords] = block

    return 1
