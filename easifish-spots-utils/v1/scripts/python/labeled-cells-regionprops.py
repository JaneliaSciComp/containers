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


def _extract_cell_region_properties(args):
    image_data, _ = imgio.open(args.image_container, args.image_dataset)
    labels, _ = imgio.open(args.labels_container, args.labels_dataset)

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

    labels_stat = regionprops(labels, intensity_image=image)
    df = pd.DataFrame(data=np.empty([len(labels_stat), 5]),
                      columns=['roi', 'area',
                               'weighted_centroid',
                               'weighted_local_centroid', 'mean_intensity'],
                      dtype=object)

    for i in range(0, len(labels_stat)):
        df.loc[i, 'roi'] = labels_stat[i].label
        df.loc[i, 'area'] = labels_stat[i].area
        df.loc[i, 'weighted_centroid'] = labels_stat[i].centroid_weighted
        df.loc[i, 'weighted_local_centroid'] = labels_stat[i].centroid_weighted_local
        df.loc[i, 'mean_intensity'] = labels_stat[i].intensity_mean

    print("Writing", args.output)
    df.to_csv(args.output, index=False)




def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    # run post processing
    _extract_cell_region_properties(args)


if __name__ == '__main__':
    _main()
