import functools
import logging
import numpy as np
import re
import scipy.ndimage as ndi
import zarr

from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client, as_completed
from .ngff.ngff_utils import (get_transformations_from_datasetpath,
                              get_first_space_axis, get_multiscales, add_new_dataset)
from toolz import partition_all
from typing import Tuple
from xarray_multiscale import windowed_mean, windowed_mode


logger = logging.getLogger(__name__)


def create_multiscale(multiscale_group: zarr.Group,
                      group_attrs: dict,
                      dataset_path: str|None,
                      dataset_pattern: str,
                      data_type: str,
                      antialiasing:bool,
                      partition_size:int,
                      skip_attrs_update:bool,
                      client: Client):
    """
    Create a multiscale pyramid in the given Zarr group.
    """
    logger.info(f'Create multiscale for dataset at {multiscale_group.path}:{dataset_path}')
    dataset_regex = re.compile(dataset_pattern)
    pyramid_attrs = get_multiscales(group_attrs)

    source_dataset_shape = group_attrs.get('dataset_shape', [])
    source_dataset_level = int(dataset_regex.match(dataset_path).group(1))
    source_dataset_scale, source_dataset_translation = get_transformations_from_datasetpath(
        pyramid_attrs, dataset_path,
        default_scale=(1,) * len(source_dataset_shape),
        default_translation=(0,) * len(source_dataset_shape),
    )
    dataset_blocksize = group_attrs.get('dataset_blocksize', [])

    def is_spatial_axis(axis:int) -> bool:
        return axis >= get_first_space_axis(pyramid_attrs, dataset_dims=len(source_dataset_shape))

    absolute_scaling_factors = np.array([pow(2, source_dataset_level)
                                        if is_spatial_axis(i) else 1
                                        for i,_ in enumerate(source_dataset_shape)])
    if source_dataset_level == 0:
        level0_translation = source_dataset_translation
        level0_scale = source_dataset_scale

    else:
        level0_scale = tuple(s / pow(2,source_dataset_level)
                             if is_spatial_axis(i) else s
                             for i,s in enumerate(source_dataset_scale))
        level0_translation = tuple(t - (pow(2,source_dataset_level-1)-0.5)
                               if is_spatial_axis(i) else t
                               for i,t in enumerate(source_dataset_translation))

    logger.info((
        f'Source level: {source_dataset_level}, '
        f'source dataset path: {dataset_path}, '
        f'shape: {source_dataset_shape} '
        f'source scaling factors: {absolute_scaling_factors} '
        f'source scale: {source_dataset_scale} '
        f'source translation: {source_dataset_translation} '
        f'level0 scale: {level0_scale} '
        f'level0 translation: {level0_translation} '
    ))

    dataset_arr = multiscale_group[dataset_path] if dataset_path else multiscale_group
    current_level_shape = dataset_shape = dataset_arr.shape

    def next_level(match):
        level = int(match.group(1))
        return match.group(0).replace(str(level), str(level + 1), 1)

    while any([dim > dataset_blocksize[i]
              for i,dim in enumerate(current_level_shape) if is_spatial_axis(i)]):
        # all spatial dimensions are larger than the corresponding block size
        new_level_path = dataset_regex.sub(next_level, dataset_path)
        new_level = int(dataset_regex.match(new_level_path).group(1))
        relative_scaling_factors = np.array([2
                                            if is_spatial_axis(i) and dim > dataset_blocksize[i]
                                            else 1
                                            for i, dim in enumerate(current_level_shape)]).astype(int)

        absolute_scaling_factors = absolute_scaling_factors * relative_scaling_factors
        current_level_scale = tuple(absolute_scaling_factors * level0_scale)
        current_level_translation = tuple([round((s0 * (dsf / 2 - 0.5)) + tr0, 3) if dsf > 1 else tr0
                                                 for (s0, dsf, tr0)
                                                 in zip(level0_scale, absolute_scaling_factors, level0_translation)])
        current_level_shape = (dataset_shape / absolute_scaling_factors).astype(int)

        logger.info((
            f'Level: {new_level}, '
            f'level dataset path: {new_level_path}, '
            f'dataset shape (l0 -> l{new_level}): {dataset_shape} -> {current_level_shape} '
            f'level downsampling factors (rel/abs): {relative_scaling_factors} / {absolute_scaling_factors} '
            f'level scale: {current_level_scale} '
            f'level translation: {current_level_translation} '
        ))

        pyramid_attrs = add_new_dataset(
            pyramid_attrs,
            new_level_path,
            scale_transform=current_level_scale,
            translation_transform=current_level_translation
        )

        logger.info((
            f'Create new dataset for level {new_level} at {new_level_path} '
            f'pyramid_attrs -> {pyramid_attrs} '
        ))

        new_dataset_arr = multiscale_group.require_dataset(
            new_level_path,
            shape=current_level_shape,
            chunks=dataset_blocksize,
            dtype=dataset_arr.dtype,
            compressor=dataset_arr.compressor,
            fill_value=dataset_arr.fill_value,
            dimension_separator='/',
            exact=True,
        )

        downsample = functools.partial(
            _downsample,
            dataset_arr,
            new_dataset_arr,
            downsampling_factors=relative_scaling_factors,
            method='mode' if data_type == 'segmentation' else 'mean',
            antialising=antialiasing,
        )

        output_chunks = normalize_chunks(new_dataset_arr.chunks, shape=new_dataset_arr.shape)
        output_slices = slices_from_chunks(output_chunks)
        partitioned_output_slices = tuple(partition_all(partition_size, output_slices))
        logger.info(f'Partition level {new_level} with {len(output_slices)} blocks into {len(partitioned_output_slices)} partitions of up to {partition_size} blocks')

        for idx, part in enumerate(partitioned_output_slices):
            logger.info(f'Process level {new_level} partition {idx} ({len(part)} blocks)')

            res = client.map(downsample, part)
            for f, r in as_completed(res, with_results=True):
                if f.cancelled():
                    exc = f.exception()
                    logger.exception(f'Level {new_level}: block processing exception: {exc}')
                    res = False
                else:
                    logger.debug(f'Level {new_level}: Finished writing blocks {r}')

            logger.info(f'Finished level {new_level} partition {idx}')

        dataset_path = new_level_path
        dataset_arr = new_dataset_arr

    if not skip_attrs_update:
        multiscale_group.attrs.update({
            'multiscales': [ pyramid_attrs ],
        })


def _downsample(input:zarr.Array,
                output:zarr.Array,
                output_coords: Tuple[slice, ...],
                downsampling_factors=(2,2,2),
                method='mean',
                antialising=False):
    """
    Downsample source to target shape using the specified method.
    """

    input_coords = tuple(_multiply_slice(s, fact) for s, fact in zip(output_coords, downsampling_factors))
    logger.debug(f'Resample block {input_coords} -> {downsampling_factors}')
    input_block = input[input_coords]
    # only downsample source_data if it is not all 0s
    if not (input_block == 0).all():
        if method == 'mode':
            # this is the method used for segmentation data
            output[output_coords] = windowed_mode(input_block, window_size=downsampling_factors)
        else:
            # this is the method used for raw image data
            if antialising:
                # blur data in chunk before downsampling to reduce aliasing of the image
                # conservative Gaussian blur coeff: 2/2.5 = 0.8
                sigma = [0 if factor == 1 else factor/2.5 for factor in downsampling_factors]
                filtered_data = ndi.gaussian_filter(input_block, sigma=sigma)
                output[output_coords] = windowed_mean(filtered_data, window_size=downsampling_factors)
            else:
                output[output_coords] = windowed_mean(input_block, window_size=downsampling_factors)
        res = True
    else:
        res = False

    del input_block
    return output_coords, res


def _multiply_slice(s:slice, factor:int):
    return slice(s.start * factor, s.stop * factor, s.step)
