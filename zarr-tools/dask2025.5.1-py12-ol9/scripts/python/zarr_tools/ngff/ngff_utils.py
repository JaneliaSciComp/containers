from ome_zarr_models.v04.image import (Dataset)


def add_new_dataset(multiscales_attrs, dataset_path, scale_transform, translation_transform):
    datasets = multiscales_attrs.get('datasets', [])
    dataset_paths = [ds.get('path') for ds in datasets if ds.get('path', None)]

    print(f'Add dataset path: {dataset_path} to {multiscales_attrs}')

    if scale_transform is not None:
        dataset = Dataset.build(path=dataset_path,
                                scale=scale_transform,
                                translation=translation_transform)

        existing_dataset = next((ds for ds in datasets if ds.get('path') == dataset_path), None)
        if existing_dataset is None:
            datasets.append(dataset.dict(exclude_none=True))

        existing_path = next((p for p in dataset_paths if p == dataset_path), None)
        if existing_path is None:
            dataset_paths.append(dataset_path)

        multiscales_attrs.update({
            'datasets': datasets,
            'paths': dataset_paths,
        })

    return multiscales_attrs


def get_first_space_axis(multiscales_attrs, dataset_dims=-1, spatial_dims=3):
    """
    Get the first space axis index from the multiscale attributes.
    """
    axes = multiscales_attrs.get('axes', None)
    if axes is not None:
        for i, axis in enumerate(axes):
            if axis.get('type') == 'space' or axis.get('name', '').lower() in ['z', 'y', 'x']:
                return i

    return dataset_dims - spatial_dims if dataset_dims > spatial_dims else 0


def get_dataset(multiscale_attrs, dataset_path):
    datasets = multiscale_attrs.get('datasets', [])
    dataset_path_comps = [c for c in dataset_path.split('/') if c]
    # lookup the dataset by path
    for ds in datasets:
        ds_path = ds.get('path', '')
        ds_path_comps = [c for c in ds_path.split('/') if c]
        print((
            f'Compare current dataset path: {ds_path} ({ds_path_comps}) '
            f'with {dataset_path} ({dataset_path_comps}) '
        ))
        if (len(ds_path_comps) <= len(dataset_path_comps) and
            tuple(ds_path_comps) == tuple(dataset_path_comps[-len(ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            print(f'Found dataset at {ds_path}')
            return ds

        return None


def get_dataset_transformations(multiscale_attrs, dataset_path, default_scale=None, default_translation=None):
    """
    Get the scale and translation transformations from the multiscale attributes for the dataset with the given path.
    """
    dataset_metadata = get_dataset(multiscale_attrs, dataset_path)

    scale, translation = default_scale, default_translation
    coord_transformations = (dataset_metadata.get('coordinateTransformations', [])
                             if dataset_metadata is not None else [])
    for t in coord_transformations:
        if t['type'] == 'scale':
            scale = t['scale']
        elif t['type'] == 'translation':
            translation = t['translation']

    return scale, translation


def get_datasets(multiscale_attrs):
    return multiscale_attrs.get('datasets', [])


def get_multiscales(attrs, index=0):
    """
    Get the multiscales attributes.
    """
    return attrs.get('multiscales', [{}])[index] # get the multiscale attributes at the specified index


def has_multiscales(attrs):
    """
    Get the multiscales attributes.
    """
    return 'multiscales' in attrs
