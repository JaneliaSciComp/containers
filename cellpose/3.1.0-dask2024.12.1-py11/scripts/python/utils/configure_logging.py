import logging
import os
import sys

from logging.config import fileConfig


def configure_logging(config_file, verbose):
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if config_file and os.path.exists(config_file):
        fileConfig(config_file)
    else:
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(stream=sys.stdout)
                            ])
    logger = logging.getLogger()
    logger.setLevel(log_level)
    return logger


def _set_cpu_resources(cpus:int):
    if cpus:
        os.environ['MKL_NUM_THREADS'] = str(cpus)
        os.environ['NUM_MKL_THREADS'] = str(cpus)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpus)
        os.environ['OPENMP_NUM_THREADS'] = str(cpus)
        os.environ['OMP_NUM_THREADS'] = str(cpus)

    return cpus
