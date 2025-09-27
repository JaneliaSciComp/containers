import argparse

from dask.distributed import (Client, LocalCluster)

from zarr_tools.configure_logging import configure_logging
from zarr_tools.dask_tools import (load_dask_config, ConfigureWorkerPlugin)
from zarr_tools.multiscale import create_multiscale
from zarr_tools.zarr_io import open_zarr


logger = None


def _define_args():
    args_parser = argparse.ArgumentParser()

    input_args = args_parser.add_argument_group("Input Arguments")
    input_args.add_argument('-i', '--input',
                             dest='input',
                             type=str,
                             help = 'input directory')
    input_args.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = 'input subpath')
    input_args.add_argument('--dataset-pattern', '--dataset_pattern',
                             dest='dataset_pattern',
                             type=str,
                             help = 'dataset pattern to extract the numeric scale level')
    input_args.add_argument('--data-type', '--data_type',
                             dest='data_type',
                             type=str,
                             default='raw',
                             help = 'data type (e.g. segmentation, raw)')
    input_args.add_argument('--antialiasing',
                             dest='antialiasing',
                             default=False,
                             action='store_true',
                             help = 'antialiasing')
    input_args.add_argument('--skip-metadata', '--skip_metadata',
                             dest='skip_metadata',
                             default=False,
                             action='store_true',
                             help = 'Skip metadata update')

    input_args.add_argument('--logging-config', '--logging_config',
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
    distributed_args.add_argument('--partition-size', '--partition_size',
                                  dest='partition_size',
                                  type=int,
                                  default=10000,
                                  help='Processing partition size')

    return args_parser


def _run_multiscale(args):
    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        dask_cluster = None
        dask_client = Client(address=args.dask_scheduler)
    else:
        # use a local asynchronous client
        dask_cluster = LocalCluster(n_workers=args.local_dask_workers,
                                    threads_per_worker=args.worker_cpus,
                                    processes=False)
        dask_client = Client(dask_cluster)

    worker_config = ConfigureWorkerPlugin(args.logging_config,
                                          worker_cpus=args.worker_cpus)
    dask_client.register_plugin(worker_config, name='WorkerConfig')

    dataset_container, dataset_attrs, dataset_path = open_zarr(
        args.input, args.input_subpath, mode='a'
    )

    partition_size = args.partition_size if args.partition_size > 0 else 1
    dataset_pattern = args.dataset_pattern if args.dataset_pattern else '.*(\\d+?)'
    create_multiscale(dataset_container, dataset_attrs, dataset_path, dataset_pattern,
                      args.data_type, args.antialiasing,
                      partition_size,
                      args.skip_metadata,
                      dask_client)

    if dask_cluster is not None:
        dask_cluster.close()
    else:
        dask_client.close()


def main():
    args_parser = _define_args()
    args = args_parser.parse_args()
    # prepare logging
    global logger
    logger = configure_logging(args.logging_config)

    # run multi-scale segmentation
    print(f'Run multiscale: {args}')
    _run_multiscale(args)


if __name__ == '__main__':
    main()