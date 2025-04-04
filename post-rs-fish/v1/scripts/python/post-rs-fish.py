import argparse
import numpy as np

import io_utils.zarr_utils as zarr_utils


def _floattuple(arg):
    if arg is not None and arg.strip():
        return tuple([float(d) for d in arg.split(',')])
    else:
        return ()


def _define_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-i', '--input',
                             dest='input',
                             type=str,
                             required=True,
                             help = "spots input file using voxel coordinates")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             required=True,
                             help = "output file in real coordinates")

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
                             type=_floattuple,
                             help = "voxel spacing")
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')



    return args_parser


def _post_process_rsfish_csv_results(args):

    image_data, image_attrs = zarr_utils.open(args.image_container, args.image_dataset)
    image_ndim = image_data.ndim

    if args.voxel_spacing:
        voxel_spacing = zarr_utils.get_voxel_spacing({}, args.voxel_spacing)
    else:
        voxel_spacing = zarr_utils.get_voxel_spacing(image_attrs, (1.,) * image_ndim)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            expansion = args.expansion_factor
        else:
            expansion = 1.
        voxel_spacing /= expansion
    else:
        voxel_spacing = np.array((1.,) * image_ndim)

    print(f"Image voxel spacing: {voxel_spacing}")

    rsfish_spots = np.loadtxt(args.input, delimiter=',', skiprows=1)
    rsfish_spots[:, :3] = rsfish_spots[:, :3] * voxel_spacing

    # Remove unnecessary columns (t,c) at indexes 3 and 4 
    rsfish_spots = np.delete(rsfish_spots, np.s_[3:5], axis=1)

    print(f'Saving {rsfish_spots.shape[0]} points in micron space to {args.output}')
    np.savetxt(args.output, rsfish_spots, delimiter=',')


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    # run post processing
    _post_process_rsfish_csv_results(args)


if __name__ == '__main__':
    _main()
