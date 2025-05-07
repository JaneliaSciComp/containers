import argparse
import numpy as np
import pandas as pd

import io_utils.imgio as imgio
from cli import floattuple
from skimage.measure import regionprops


def _define_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--labels-container',
                             dest='labels_container',
                             type=str,
                             required=False,
                             help = "path to the labels container")
    args_parser.add_argument('--labels-subpath', '--labels-dataset',
                             dest='labels_dataset',
                             type=str,
                             required=False,
                             help = "path to the labels container")

    args_parser.add_argument('--image-container',
                             dest='image_container',
                             type=str,
                             help = "image container")
    args_parser.add_argument('--image-subpath', '--image-dataset',
                             dest='image_dataset',
                             type=str,
                             help = "image subpath")

    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=floattuple,
                             help = "voxel spacing")
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')

    args_parser.add_argument('--dapi-subpath', '--dapi-dataset',
                             dest='dapi_dataset',
                             type=str,
                             help = "DAPI image subpath")
    args_parser.add_argument('--bleed-subpath', '--bleed-dataset',
                             dest='bleed_dataset',
                             type=str,
                             help = "Bleed image subpath")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             required=True,
                             help = "output file in real coordinates")

    return args_parser


def _extract_spots_region_properties(args):
    image_data, image_attrs = imgio.open(args.image_container, args.image_dataset)
    labels_zarr, _ = imgio.open(args.labels_container, args.labels_dataset)

    if args.voxel_spacing:
        voxel_spacing = imgio.get_voxel_spacing({}, args.voxel_spacing, as_zyx=True)
    else:
        # get voxel spacing from the labels attributes
        voxel_spacing = imgio.get_voxel_spacing(image_attrs, args.voxel_spacing, as_zyx=True)

    if voxel_spacing is None:
        if args.expansion_factor > 0:
            expansion = args.expansion_factor
        else:
            expansion = 1.
        voxel_spacing /= expansion
    else:
        image_ndim = len(image_data.shape)
        voxel_spacing = np.array((1.,) * image_ndim)

    image = image_data[...]

    if (args.bleed_dataset is not None and
        args.dapi_dataset is not None and
        args.bleed_dataset == args.image_dataset):
        dapi_data, _ = imgio.open(args.image_container, args.dapi_dataset)
        dapi = dapi_data[...]
        lo = np.percentile(np.ndarray.flatten(dapi), 99.5)
        bg_dapi = np.percentile(np.ndarray.flatten(dapi[dapi != 0]), 1)
        bg_img = np.percentile(np.ndarray.flatten(image[image != 0]), 1)
        dapi_factor = np.median((image[dapi > lo] - bg_img) /
                                (dapi[dapi > lo] - bg_dapi))
        image = np.maximum(0, image - bg_img - dapi_factor * (dapi - bg_dapi)).astype('float32')
        print(f'Corrected bleed dataset {args.bleed_dataset} {image.shape} image')
        print('bleed_through:', dapi_factor)
        print('DAPI background:', bg_dapi)
        print('bleed_through channel background:', bg_img)

    labels = labels_zarr[...]
    labels_stats = regionprops(labels, intensity_image=image, spacing=voxel_spacing)

    df = pd.DataFrame(data=np.empty([len(labels_stats), 5]),
                      columns=['roi', 'area',
                               'mean_intensity',
                               'weighted_centroid',
                               'weighted_local_centroid',
                               ],
                      dtype=object)

    for i in range(0, len(labels_stats)):
        df.loc[i, 'roi'] = labels_stats[i].label
        df.loc[i, 'area'] = labels_stats[i].area
        df.loc[i, 'mean_intensity'] = labels_stats[i].intensity_mean
        df.loc[i, 'weighted_centroid'] = labels_stats[i].centroid_weighted
        df.loc[i, 'weighted_local_centroid'] = labels_stats[i].centroid_weighted_local

    print("Writing", args.output)
    df.to_csv(args.output, index=False)




def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    # run post processing
    _extract_spots_region_properties(args)


if __name__ == '__main__':
    _main()
