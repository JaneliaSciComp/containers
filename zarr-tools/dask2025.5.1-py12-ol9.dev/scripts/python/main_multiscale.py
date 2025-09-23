import argparse

from dask.distributed import (Client, LocalCluster)

from zarr_tools.zarr_io import open_zarr
from zarr_tools.multiscale import create_multiscale
from zarr_tools.dask_tools import (load_dask_config, ConfigureWorkerPlugin)



def _define_args():
    args_parser = argparse.ArgumentParser()

    input_args = args_parser.add_argument_group("Input Arguments")
    input_args.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             help = "input directory")
    input_args.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "input subpath")
    input_args.add_argument('--dataset-pattern',
                             dest='dataset_pattern',
                             type=str,
                             help = "dataset pattern")
    input_args.add_argument('--data-type',
                             dest='data_type',
                             type=str,
                             default='raw',
                             help = "data type (e.g. segmentation, raw)")

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


def _run_multiscale(args):
    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        dask_client = Client(address=args.dask_scheduler)
    else:
        # use a local asynchronous client
        dask_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                          threads_per_worker=args.worker_cpus))

    worker_config = ConfigureWorkerPlugin(worker_cpus=args.worker_cpus)
    dask_client.register_plugin(worker_config, name='WorkerConfig')

    dataset_container, dataset_attrs, dataset_path = open_zarr(
        args.input, args.input_subpath, mode='a'
    )

    dataset_pattern = args.dataset_pattern if args.dataset_pattern else '.*(\\d+?)'
    create_multiscale(dataset_container, dataset_attrs, dataset_path, dataset_pattern, args.data_type, dask_client)

    dask_client.close()



if __name__ == '__main__':
    args_parser = _define_args()
    args = args_parser.parse_args()

    # run multi-scale segmentation
    print(f'Run multiscale: {args}')
    _run_multiscale(args)
