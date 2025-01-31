import logging
import numpy as np
import yaml
import pydantic.v1.utils as pu
from scipy.ndimage import gaussian_filter


logger = logging.getLogger(__name__)

default_preprocessing_params_str="""
unsharp:
    sigma_one: 1.0
    weight: 0.1
    iterations: 5
    sigma_two: 0.1
"""


def get_preprocessing_steps(steps, preprocessing_config_file):
    if not steps:
        return []

    _preprocessing_methods = {
        'unsharp': _unsharp
    }

    preprocessing_steps = []
    preprocessing_config = _get_preprocessing_config(preprocessing_config_file)
    for step in steps:
        step_method = _preprocessing_methods.get(step)
        if step_method is not None:
            step_params = preprocessing_config.get(step)
            if step_params is not None:
                logger.info(f'Add preprocessing step: {step}:{step_params}')
                preprocessing_steps = preprocessing_steps.append((step_method, step_params))

    return preprocessing_steps


def _get_preprocessing_config(preprocessing_config_file):
    default_config = yaml.safe_load(default_preprocessing_params_str)
    logger.info(f'Default preprocessing config: {default_config}')
    if preprocessing_config_file:
        with open(preprocessing_config_file) as f:
            external_config = yaml.safe_load(f)
            logger.info((
                'Read external config from '
                f'{preprocessing_config_file}: {external_config}'
            ))
            config = pu.deep_update(default_config, external_config)
            logger.info(f'Final config {config}')
    else:
        config = default_config


def _unsharp(image, sigma_one, weight, iterations, sigma_two):
    image = image.astype(np.float32)
    for i in range(iterations):
        high_frequency_image = image - gaussian_filter(image, sigma_one)
        image = (1. - weight) * image + weight * high_frequency_image

    image = gaussian_filter(image, sigma_two)
    image[image < 0] = 0
    image[image > 65500] = 65500
    return np.round(image).astype(np.uint16)
