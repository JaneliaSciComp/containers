import argparse
import numpy as np

from dask.distributed import (Client, LocalCluster)
from io_utils import read_utils
from pathlib import Path

from .configure_dask import (ConfigureWorkerPlugin, load_dask_config)
from .configure_fishspots import get_fishspots_config
from .distribute import distributed_spot_detection


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _floattuple(arg):
    if arg is not None and arg.strip():
        return tuple([float(d) for d in arg.split(',')])
    else:
        return ()


def _define_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--input', 
                             type=str,
                             required=True,
                             help='Path to the input container')
    args_parser.add_argument('--input_subpath',
                             type=str,
                             default=None,
                             help='Optional dataset subpath within the input')
    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=_floattuple,
                             help='Image voxel spacing')
    args_parser.add_argument('--timeindex',
                             type=int,
                             default=None,
                             help='Time index to process (if applicable)')
    args_parser.add_argument('--channels', '--included-channels',
                             dest='channels',
                             type=int,
                             nargs='+',
                             default=[],
                             help='List of channel indices to process')
    args_parser.add_argument('--excluded-channels',
                             dest='excluded_channels',
                             type=int,
                             nargs='+',
                             default=[],
                             help='List of channel indices to process')
    args_parser.add_argument('--output',
                             type=str,
                             required=True,
                             help='Path to the output file')
    args_parser.add_argument('--apply-voxel-spacing', '--apply_voxel_spacing',
                             dest='apply_voxel_spacing',
                             action='store_true',
                             default=False,
                             help="Apply voxel spacing")

    args_parser.add_argument('--dask-scheduler',
                             dest='dask_scheduler',
                             type=str,
                             default=None,
                             help='Run with distributed scheduler')
    args_parser.add_argument('--dask-config', dest='dask_config',
                             type=str, default=None,
                             help='YAML file containing dask configuration')
    args_parser.add_argument('--local-dask-workers', '--local_dask_workers',
                             dest='local_dask_workers',
                             type=int,
                             help='Number of workers when using a local cluster')
    args_parser.add_argument('--worker-cpus',
                             dest='worker_cpus',
                             type=int, default=0,
                             help='Number of cpus allocated to a dask worker')

    args_parser.add_argument('--blocksize',
                             type=_inttuple,
                             default=(),
                             help='Block size as [x,y,z] size')

    args_parser.add_argument('--fishspots-config', '--fishspots_config',
                             dest='fishspots_config',
                             type=str,
                             help='Fishspots config yaml file')
    args_parser.add_argument('--psf-retries', '--psf_retries',
                             dest='psf_retries',
                             type=int,
                             default=3,
                             help='PSF retries')
    args_parser.add_argument('--intensity-threshold', '--intensity_threshold',
                             dest='intensity_threshold',
                             type=int,
                             default=0,
                             help='Intensity threshold for spot detection')
    return args_parser


def _write_spots(spots, csvfilename):
    # x,y,z -> z,y,x
    spots[:, :3] = spots[:, :3][:, ::-1]

    header = 'x,y,z,t,c,intensity'
    fmt = ['%.4f', '%.4f', '%.4f', '%d', '%d', '%.4f']

    print(f'Write spots to {csvfilename}')
    output_path = Path(csvfilename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csvfilename, spots, delimiter=',', header=header, fmt=fmt)


def _main():
    # parse CLI args
    args_parser = _define_args()
    args = args_parser.parse_args()

    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        cluster_client = Client(address=args.dask_scheduler)
    else:
        cluster_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                             threads_per_worker=args.worker_cpus))

    worker_config = ConfigureWorkerPlugin(worker_cpus=args.worker_cpus)
    cluster_client.register_plugin(worker_config, name='WorkerConfig')

    image_data, image_attrs = read_utils.open(args.input, args.input_subpath)

    if args.voxel_spacing:
        voxel_spacing = args.voxel_spacing[::-1]
    else:
        voxel_spacing = read_utils.get_voxel_spacing(image_attrs)

    if args.blocksize:
        # convert the x,y,z input block size to z,y,x
        processing_blocksize = args.blocksize[::-1]
    else:
        processing_blocksize = image_data.shape[-3:]

    fishspots_config = get_fishspots_config(args.fishspots_config)
    white_tophat_args=fishspots_config.get('white_tophat_args', {})
    psf_estimation_args=fishspots_config.get('psf_estimation_args', {})
    deconvolution_args=fishspots_config.get('deconvolution_args', {})
    spot_detection_args=fishspots_config.get('spot_detection_args', {})

    spots, _ = distributed_spot_detection(
        image_data,
        args.timeindex,
        args.channels,
        set(args.excluded_channels) if args.excluded_channels else set(),
        processing_blocksize,
        cluster_client,
        white_tophat_args=white_tophat_args,
        psf_estimation_args=psf_estimation_args,
        deconvolution_args=deconvolution_args,
        spot_detection_args=spot_detection_args,
        intensity_threshold=args.intensity_threshold,
        psf_retries=args.psf_retries,
    )

    if args.apply_voxel_spacing:
        print(f'Apply voxel spacing: {voxel_spacing}')
        spots[:, :3] = spots[:, :3] * voxel_spacing
    
    _write_spots(spots, args.output)

if __name__ == '__main__':
    _main()