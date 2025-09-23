import functools
import numpy as np
import re
import zarr

from dask.distributed import Client
from typing import Dict




def combine_arrays(input_zarrays: Dict[int,zarr.Array],
                   client: Client):
    """
    Combine arrays
    """




