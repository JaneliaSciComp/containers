import os
import yaml

from dask.distributed import Worker
from distributed.diagnostics.plugin import WorkerPlugin
from flatten_json import flatten

from .configure_logging import configure_logging


class ConfigureWorkerPlugin(WorkerPlugin):
    def __init__(self, logging_config, worker_cpus=0):
        self.logging_config = logging_config
        self.worker_cpus = worker_cpus

    def setup(self, worker: Worker):
        self.logger = configure_logging(self.logging_config)
        _set_cpu_resources(self.worker_cpus)

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        pass

    def release_key(self, key: str, state: str, cause: str | None, reason: None, report: bool):
        pass


def load_dask_config(config_file):
    if (config_file):
        import dask.config

        print(f'Use dask config: {config_file}', flush=True)
        
        with open(config_file) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)


def _set_cpu_resources(cpus:int):
    if cpus:
        os.environ['MKL_NUM_THREADS'] = str(cpus)
        os.environ['NUM_MKL_THREADS'] = str(cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpus)
        os.environ['OPENMP_NUM_THREADS'] = str(cpus)
        os.environ['OMP_NUM_THREADS'] = str(cpus)

    return cpus
