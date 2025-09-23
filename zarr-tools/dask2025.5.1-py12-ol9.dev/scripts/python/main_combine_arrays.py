import argparse

from dask.distributed import (Client, LocalCluster)
from dataclasses import dataclass

from zarr_tools.combine_arrays import combine_arrays
from zarr_tools.dask_tools import (load_dask_config, ConfigureWorkerPlugin)
from zarr_tools.zarr_io import (open_zarr,create_zarr_array)


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
    input_args.add_argument('--output-subpath',
                             dest='output_subpath',
                             type=str,
                             help = "Output subpath")
    input_args.add_argument('--output_chunks',
                            type=_inttuple,
                            dest='output_chunks',
                            metavar='X,Y,Z',
                            default=(128,128,128),
                            help='Spatial output chunks')
    input_args.add_argument('--output_type',
                            type=str,
                            dest='output_type',
                            help='Output type')
    input_args.add_argument('--overwrite',
                            dest='overwrite',
                            action='store_true',
                            help='Overwrite container if it exists')

    input_args.add_argument('--compressor',
                            default='zstd',
                            help='Zarr array compression algorithm')
    
    input_args.add_argument('--array-params',
                            nargs='+',
                            metavar='SOURCEPATH:SOURCESUBPATH:TARGETCH:TARGETTP',
                            default=[None, None, None, None],
                            type=_arrayparams,
                            dest='array_params',
                            help='Input array argument')

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

    return args_parser


def _run_combine_arrays(args):
    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        dask_client = Client(address=args.dask_scheduler)
    else:
        # use a local asynchronous client
        dask_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                          threads_per_worker=args.worker_cpus))

    worker_config = ConfigureWorkerPlugin(worker_cpus=args.worker_cpus)
    dask_client.register_plugin(worker_config, name='WorkerConfig')

    input_zarrays = []
    spatial_shape = ()
    max_ch = 0
    max_tp = None
    errors_found = []
    output_type = args.output_type
    for ap in args.array_params:
        array_container = ap.sourcePath if ap.sourcePath else args.input
        zgroup, zattrs, zsubpath = open_zarr(array_container, ap.sourceSubpath, mode='r')
        zarray = zgroup[zsubpath]
        print(f'!!!!!! {ap.sourceSubpath}: {zsubpath} {zarray.shape}')

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
        
        input_zarrays.append((zarray, ap.targetCh, ap.targetTp))

    xyz_output_chunks = args.output_chunks if args.output_chunks else (128,) * 3

    if max_tp is not None:
        output_shape = (max_tp+1, max_ch+1) + spatial_shape
        output_chunks = (1,1) + xyz_output_chunks[::-1]
    else:
        output_shape = (max_ch+1,) + spatial_shape
        output_chunks = (1,) + xyz_output_chunks[::-1]

    if len(errors_found) > 0:
        print(f'Errors found: {errors_found}')
    else:
        print(f'Create output {args.output}:{args.output_subpath}:{output_shape}:{output_chunks}:{output_type}')
        output_zarray = create_zarr_array(
            args.output,
            args.output_subpath,
            output_shape,
            output_chunks,
            output_type,
            overwrite=args.overwrite,
        )
        combine_arrays(input_zarrays, output_zarray, dask_client)

    dask_client.close()



if __name__ == '__main__':
    args_parser = _define_args()
    args = args_parser.parse_args()

    # run multi-scale segmentation
    print(f'Combine arrays: {args}')
    _run_combine_arrays(args)
