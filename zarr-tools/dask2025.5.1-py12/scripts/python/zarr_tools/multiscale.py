import functools
import numpy as np
import re
import zarr

from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client
from .ngff.ngff_utils import (get_dataset_transformations, get_first_space_axis,
                              get_multiscales, add_new_dataset)
from xarray_multiscale import windowed_mean, windowed_mode


def create_multiscale(multiscale_group: zarr.Group,
                      group_attrs: dict,
                      dataset_path: str,
                      dataset_pattern: str,
                      data_type: str,
                      client: Client):
    """
    Create a multiscale pyramid in the given Zarr group.
    """
    dataset_regex = re.compile(dataset_pattern)
    pyramid_attrs = get_multiscales(group_attrs)

    source_dataset_shape = group_attrs.get('dataset_shape', [])
    source_dataset_level = int(dataset_regex.match(dataset_path).group(1))
    source_dataset_scale, source_dataset_translation = get_dataset_transformations(
        pyramid_attrs, dataset_path,
        default_scale=(1,) * len(source_dataset_shape),
        default_translation=(0,) * len(source_dataset_shape),
    )
    dataset_blocksize = group_attrs.get('dataset_blocksize', [])

    def is_spatial_axis(axis:int) -> bool:
        return axis >= get_first_space_axis(pyramid_attrs, dataset_dims=len(source_dataset_shape))

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

    print((
        f'Source level: {source_dataset_level}, '
        f'source dataset path: {dataset_path}, '
        f'shape: {source_dataset_shape} '
        f'source scale: {source_dataset_scale} '
        f'source translation: {source_dataset_translation} '
        f'level0 scale: {level0_scale} '
        f'level0 translation: {level0_translation} '
    ))

    dataset_arr = multiscale_group[dataset_path] if dataset_path else multiscale_group
    dataset_shape = dataset_arr.shape

    def next_level(match):
        level = int(match.group(1))
        return match.group(0).replace(str(level), str(level + 1), 1)

    while all([dim > dataset_blocksize[i] // 2
               for i,dim in enumerate(dataset_shape) if is_spatial_axis(i)]):
        # all spatial dimensions are larger than the corresponding block size
        new_level_path = dataset_regex.sub(next_level, dataset_path)
        new_level = int(dataset_regex.match(new_level_path).group(1))
        new_level_shape = np.array([dim // 2
                                      if is_spatial_axis(i)
                                      else dim for i, dim in enumerate(dataset_shape)]).astype(int)
        new_level_scale = tuple(s * pow(2, new_level)
                                if is_spatial_axis(i) else s
                                for i,s in enumerate(level0_scale) )
        new_level_translation = tuple(t + (pow(2,new_level-1)-0.5)
                               if is_spatial_axis(i) else t
                               for i,t in enumerate(level0_translation))

        downsampling_factors = tuple(int(dataset_shape[i] // new_level_shape[i])
                                     for i, _ in enumerate(dataset_shape))
        print((
            f'Level: {new_level}, '
            f'level dataset path: {new_level_path}, '
            f'level dataset shape: {new_level_shape} '
            f'downsampling factors: {downsampling_factors} '
            f'level scale: {new_level_scale} '
            f'level translation: {new_level_translation} '
        ))

        pyramid_attrs = add_new_dataset(
            pyramid_attrs,
            new_level_path,
            scale_transform=new_level_scale,
            translation_transform=new_level_translation
        )

        print(f'Create new dataset for level {new_level} at {new_level_path}')

        print(f'!!!!! ATTRS for {new_level} -> {pyramid_attrs}')

        new_dataset_arr = multiscale_group.require_dataset(
            new_level_path,
            shape=new_level_shape,
            chunks=dataset_blocksize,
            dtype=dataset_arr.dtype,
            compressor=dataset_arr.compressor,
            fill_value=dataset_arr.fill_value,
        )

        output_chunks = normalize_chunks(new_dataset_arr.chunks, shape=new_dataset_arr.shape)
        output_slices = slices_from_chunks(output_chunks)

        downsample = functools.partial(
            _downsample,
            dataset_arr,
            new_dataset_arr,
            downsampling_factors=downsampling_factors,
            method='mode' if data_type == 'segmentation' else 'mean'
        )

        res = client.map(downsample, output_slices)
        client.gather(res)

        dataset_arr = new_dataset_arr
        dataset_shape = new_level_shape
        dataset_path = new_level_path

    print('!!!!! UPDATE GROUP ATTRS ', multiscale_group.attrs.asdict() , ' -> ', pyramid_attrs)
    multiscale_group.attrs.update({
        'multiscales': [ pyramid_attrs ],
    })
    print('!!!!! AFTER UPDATE GROUP ATTRS ', multiscale_group.attrs.asdict())

    return None


def _downsample(input, output, output_coords, downsampling_factors=(2,2,2), method='mean'):
    """
    Downsample source to target shape using the specified method.
    """

    input_coords = tuple(_multiply_slice(s, fact) for s, fact in zip(output_coords, downsampling_factors))
    input_block = input[input_coords]

    if not (input_block == 0).all():
        if method == 'mode':
            output_block = windowed_mode(input_block, window_size=downsampling_factors)
        else:
            output_block = windowed_mean(input_block, window_size=downsampling_factors)

        output[output_coords] = output_block
        return 1

    return 0


def _multiply_slice(s, factor):
    return slice(s.start * factor, s.stop * factor, s.step)
