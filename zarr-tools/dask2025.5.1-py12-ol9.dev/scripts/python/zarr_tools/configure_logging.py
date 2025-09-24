import logging
import sys

from logging.config import fileConfig


def configure_logging(config_file):
    if config_file:
        print(f'Configure logging using {config_file}')
        fileConfig(config_file)
    else:
        print('Configure logging using basic config')
        log_level = logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=log_level,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(stream=sys.stdout)
                            ])
    return logging.getLogger()
