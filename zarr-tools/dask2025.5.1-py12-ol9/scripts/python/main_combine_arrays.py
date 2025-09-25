import argparse

from dask.distributed import (Client, LocalCluster)
from dataclasses import dataclass

from zarr_tools.ngff.ngff_utils import (get_axes, get_multiscales, get_voxel_spacing)
from zarr_tools.combine_arrays import combine_arrays
from zarr_tools.configure_logging import configure_logging
from zarr_tools.dask_tools import (load_dask_config, ConfigureWorkerPlugin)
from zarr_tools.zarr_io import (open_zarr,create_zarr_array)


logger = None


@dataclass(frozen=True)
class ZArrayParams:
    sourcePath: str
    sourceSubpath: str
    targetCh: int
    targetTp: int|None


def _arrayparams(s: str):
    svalues = s.split(':', 3)
    sourcePath = svalues[0] if len(svalues) > 0 else ''
    sourceSubpath = svalues[1] if len(svalues) > 1 else ''
    stargetCh = svalues[2] if len(svalues) > 2 else ''
    stargetTp = svalues[3] if len(svalues) > 3 else ''
    try:
        targetCh = int(stargetCh) if stargetCh else 0
    except ValueError:
        raise argparse.ArgumentTypeError(f'Invalid target channel in input array arg: {s}')
    try:
        targetTp = int(stargetTp) if stargetTp else None
    except ValueError:
        raise argparse.ArgumentTypeError(f'Invalid target timepoint in input array arg: {s}')

    return ZArrayParams(sourcePath, sourceSubpath, targetCh, targetTp)


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _define_args():
    args_parser = argparse.ArgumentParser()

    input_args = args_parser.add_argument_group("Input Arguments")
    input_args.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             help = "Default input container directory")
    input_args.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "input subpath")
    input_args.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             help = "Output container directory")
    input_args.add_argument('--voxel-spacing',
                            type=_inttuple,
                            dest='voxel_spacing',
                            metavar='X,Y,Z',
                            default=(1, 1, 1),
                            help='Spatial output chunks')
    input_args.add_argument('--output-subpath',
                             dest='output_subpath',
                             type=str,
                             help = "Output subpath")
    input_args.add_argument('--output-chunks',
                            type=_inttuple,
                            dest='output_chunks',
                            metavar='X,Y,Z',
                            default=(128, 128, 128),
                            help='Spatial output chunks')
    input_args.add_argument('--output_type',
                            type=str,
                            dest='output_type',
                            help='Output type')
    input_args.add_argument('--overwrite',
                            dest='overwrite',
                            default=False,
                            action='store_true',
                            help='Overwrite container if it exists')

    input_args.add_argument('--compressor',
                            default='zstd',
                            help='Zarr array compression algorithm')
    input_args.add_argument('--compression-level',
                            dest='compression_level',
                            type=int,
                            default=5,
                            help='Zarr array compression level')
    
    input_args.add_argument('--array-params',
                            nargs='+',
                            metavar='SOURCEPATH:SOURCESUBPATH:TARGETCH:TARGETTP',
                            default=[None, None, None, None],
                            type=_arrayparams,
                            dest='array_params',
                            help='Input array argument')

    input_args.add_argument('--skip-ome-metadata',
                            dest='skip_ome_metadata',
                            default=False,
                            action='store_true',
                            help='If set skip creation of the OME metadata')

    input_args.add_argument('--logging-config',
                            dest='logging_config',
                            type=str,
                            help='Logging configuration')

    distributed_args = args_parser.add_argument_group("Distributed Arguments")
    distributed_args.add_argument('--dask-scheduler', '--dask_scheduler',
                                  dest='dask_scheduler',
                                  type=str, default=None,
                                  help='Run with distributed scheduler')
    distributed_args.add_argument('--dask-config', '--dask_config',
                                  dest='dask_config',
                                  type=str, default=None,
                                  help='Dask configuration yaml file')
    distributed_args.add_argument('--local-dask-workers', '--local_dask_workers',
                                  dest='local_dask_workers',
                                  type=int, default=1,
                                  help='Number of workers when using a local cluster')
    distributed_args.add_argument('--worker-cpus', '--worker_cpus',
                                  dest='worker_cpus',
                                  type=int,
                                  help='Number of cpus allocated to a dask worker')
    distributed_args.add_argument('--partition-size',
                                  dest='partition_size',
                                  type=int,
                                  default=100000,
                                  help='Processing partition size')

    return args_parser


def _run_combine_arrays(args):
    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        dask_client = Client(address=args.dask_scheduler)
    else:
        # use a local asynchronous client
        dask_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                          threads_per_worker=args.worker_cpus))

    worker_config = ConfigureWorkerPlugin(args.logging_config,
                                          worker_cpus=args.worker_cpus)
    dask_client.register_plugin(worker_config, name='WorkerConfig')

    input_zarrays = []
    spatial_shape = ()
    max_ch = 0
    max_tp = None
    errors_found = []
    output_type = args.output_type
    voxel_spacing = None
    axes = None
    for ap in args.array_params:
        array_container = ap.sourcePath if ap.sourcePath else args.input
        zgroup, zattrs, zsubpath = open_zarr(array_container, ap.sourceSubpath, mode='r')
        zarray = zgroup[zsubpath] if zsubpath else zgroup

        current_voxel_spacing = get_voxel_spacing(zattrs)
        if voxel_spacing is None:
            voxel_spacing = current_voxel_spacing
        elif voxel_spacing != current_voxel_spacing:
            logger.warning(f'Voxel spacing for {current_voxel_spacing} differs from the first found: {voxel_spacing}')

        if axes is None:
            axes = get_axes(get_multiscales(zattrs))

        if not output_type:
            output_type = zarray.dtype

        if spatial_shape == ():
            spatial_shape = zarray.shape
        else:
            if spatial_shape != zarray.shape:
                errors_found.append(f'All zarr arrays must have the same spatial dimensions: {spatial_shape} - {array_container}:{ap.sourceSubpath} has shape {zarray.shape}')
        if ap.targetCh > max_ch:
            max_ch = ap.targetCh

        if ap.targetTp is not None:
            if max_tp is None:
                max_tp = ap.targetTp
            elif ap.targetTp > max_tp:
                max_tp = ap.targetTp
        
        input_zarrays.append((array_container, zsubpath, zarray, ap.targetCh, ap.targetTp))

    xyz_output_chunks = args.output_chunks if args.output_chunks else (128,) * 3

    if max_tp is not None:
        output_shape = (max_tp+1, max_ch+1) + spatial_shape
        output_chunks = (1,1) + xyz_output_chunks[::-1]
    else:
        output_shape = (max_ch+1,) + spatial_shape
        output_chunks = (1,) + xyz_output_chunks[::-1]

    if len(errors_found) > 0:
        logger.error(f'Errors found: {errors_found}')
    else:
        if voxel_spacing is None:
            voxel_spacing = list(args.voxel_spacing[::-1])

        if args.skip_ome_metadata:
            ome_metadata = {}
        else:
            ome_metadata = _create_ome_metadata(args.output_subpath,
                                                axes,
                                                voxel_spacing, 
                                                (4 if max_tp is None else 5))
        logger.info((
            f'Create output {args.output}:{args.output_subpath}:{output_shape}:{output_chunks}:{output_type} '
            f'OME metadata: {ome_metadata}'
        ))
        output_zarray = create_zarr_array(
            args.output,
            args.output_subpath,
            output_shape,
            output_chunks,
            output_type,
            compressor=args.compressor,
            compression_level=args.compression_level,
            overwrite=args.overwrite,
            parent_array_attrs=ome_metadata
        )
        logger.info(f'Combine {input_zarrays}')
        combine_arrays(input_zarrays, output_zarray, dask_client,
                       partition_size=args.partition_size)
        logger.info(f'Finished combining all arrays into {args.output}:{args.output_subpath}!')

    dask_client.close()


def _create_ome_metadata(dataset_path, axes, voxel_spacing, final_ndims, default_unit='um'):
    scale = ([1] if final_ndims == 4 else [1, 1]) + voxel_spacing
    translation = [0] * final_ndims
    if axes is None:
        multiscale_axes = [
            {
                "name": "z",
                "type": "space",
                "unit": default_unit,
            },
            {
                "name": "y",
                "type": "space",
                "unit": default_unit,
            },
            {
                "name": "x",
                "type": "space",
                "unit": default_unit,
            },
        ]
    else:
        multiscale_axes = axes[-3:]

    multiscale_axes.insert(0, {
        "name": "c",
        "type": "channel",
    })

    if final_ndims > 4:
        multiscale_axes.insert(0, {
            "name": "c",
            "type": "time",
        })

    dataset = {
        'path': dataset_path,
        'coordinateTransformations': [
            {
                'type': 'scale',
                'scale': scale,
            },
            {
                'type': 'translation',
                'translation' : translation
            }
        ]
    }
    multiscales = {
        'multiscales': [
            {
                'axes': multiscale_axes,
                'datasets': [
                    dataset
                ],
                'version': '0.4',
                'name': '/',
            }
        ]
    }

    return multiscales


def main():
    args_parser = _define_args()
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config)

    # run multi-scale segmentation
    logger.info(f'Combine arrays: {args}')
    _run_combine_arrays(args)


if __name__ == '__main__':
    main()