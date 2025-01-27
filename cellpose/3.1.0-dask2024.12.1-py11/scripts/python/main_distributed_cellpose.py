import os
import sys
import traceback

import io_utils.read_utils as read_utils
import io_utils.write_utils as write_utils

from cellpose import version_str as cellpose_version
from cellpose.cli import get_arg_parser
from dask.distributed import (Client, LocalCluster)
from flatten_json import flatten

from altcontrib.distributed_segmentation import distributed_eval as eval_with_simple_merge
from contrib.distributed_segmentation import distributed_eval as eval_with_shrink_labels_merge

from utils.configure_logging import (configure_logging)
from utils.configure_dask import (load_dask_config, ConfigureWorkerPlugin)

def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _define_args():
    args_parser = get_arg_parser()
    args_parser.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             help = "input directory")
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "input subpath")

    args_parser.add_argument('--mask',
                             dest='mask',
                             type=str,
                             help = "mask directory")
    args_parser.add_argument('--mask-subpath', '--mask_subpath',
                             dest='mask_subpath',
                             type=str,
                             help = "mask subpath")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             help = "output file")
    args_parser.add_argument('--output-subpath', '--output_subpath',
                             dest='output_subpath',
                             type=str,
                             help = "output subpath")
    args_parser.add_argument('--output-chunk-size', '--output_chunk_size',
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size as a single int')
    args_parser.add_argument('--output-blocksize', '--output_blocksize',
                             dest='output_blocksize',
                             type=_inttuple,
                             help='Output chunk size as a tuple (x,y,z).')

    args_parser.add_argument('--working-dir',
                             dest='working_dir',
                             default='.',
                             type=str,
                             help = "output file")

    args_parser.add_argument('--process-blocksize', '--process_blocksize',
                             dest='process_blocksize',
                             type=_inttuple,
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--blocks-overlaps', '--blocks_overlaps',
                             dest='blocks_overlaps',
                             type=_inttuple,
                             help='Blocks overlaps as a tuple (x,y,z).')
    args_parser.add_argument('--eval-channels', '--eval_channels',
                             dest='eval_channels',
                             type=_inttuple,
                             help='Cellpose channels: 0,0 - gray images')

    distributed_args = args_parser.add_argument_group("Distributed Arguments")
    distributed_args.add_argument('--dask-scheduler', '--dask_scheduler',
                                  dest='dask_scheduler',
                                  type=str, default=None,
                                  help='Run with distributed scheduler')
    distributed_args.add_argument('--dask-config', '--dask_config',
                                  dest='dask_config',
                                  type=str, default=None,
                                  help='Dask configuration yaml file')
    distributed_args.add_argument('--worker-cpus', dest='worker_cpus',
                                  type=int, default=0,
                                  help='Number of cpus allocated to a dask worker')
    distributed_args.add_argument('--max-cellpose-tasks', '--max_cellpose_tasks',
                                  dest='max_cellpose_tasks',
                                  type=int, default=-1,
                                  help='Max dask cellpose tasks')
    distributed_args.add_argument('--device', required=False, default='0', type=str,
                                  dest='device',
                                  help='which device to use, use an integer for torch, or mps for M1')    
    distributed_args.add_argument('--models-dir', dest='models_dir',
                                  type=str,
                                  help='cache cellpose models directory')
    distributed_args.add_argument('--model', dest='segmentation_model',
                                  type=str,
                                  default='cyto',
                                  help='segmentation model')
    distributed_args.add_argument('--iou-threshold', '--iou_threshold',
                                  dest='iou_threshold',
                                  type=float,
                                  default=0,
                                  help='Intersection over union threshold')
    distributed_args.add_argument('--iou-depth', '--iou_depth',
                                  dest='iou_depth',
                                  type=int,
                                  default=1,
                                  help='Intersection over union depth')
    distributed_args.add_argument('--save-intermediate-labels',
                                  action='store_true',
                                  dest='save_intermediate_labels',
                                  default=False,
                                  help='Save intermediate labels as zarr')
    distributed_args.add_argument('--shrink-labels-to-merge',
                                  action='store_true',
                                  dest='with_shrink_labels_merge',
                                  default=False,
                                  help='Shrink labels to merge')
    
    distributed_args.add_argument('--logging-config', dest='logging_config',
                                  type=str,
                                  help='Logging configuration')

    return args_parser


def _run_segmentation(args):
    load_dask_config(args.dask_config)
    if args.models_dir is not None:
        models_dir = args.models_dir
    elif os.environ['CELLPOSE_LOCAL_MODELS_PATH']:
        models_dir = os.environ['CELLPOSE_LOCAL_MODELS_PATH']
    else:
        models_dir = None

    worker_config = ConfigureWorkerPlugin(models_dir,
                                          args.logging_config,
                                          args.verbose,
                                          worker_cpus=args.worker_cpus)

    if args.dask_scheduler:
        dask_client = Client(address=args.dask_scheduler)
        dask_client.register_plugin(worker_config, name='WorkerConfig')
    else:
        # use a local asynchronous client
        dask_client = Client(LocalCluster())


    image_data, image_attrs = read_utils.open(args.input, args.input_subpath)
    image_ndim = image_data.ndim
    image_shape = image_data.shape
    image_dtype = image_data.dtype
    image_data = None

    logger.info(f'Image data shape/dim/dtype: {image_shape}, {image_ndim}, {image_dtype}')
    
    if args.output:
        output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

        if (args.output_blocksize is not None and
            len(args.output_blocksize) == image_ndim):
            zyx_blocksize = args.output_blocksize[::-1] # make it zyx
            output_blocks = tuple([d if d > 0 else image_shape[di] 
                                   for di,d in enumerate(zyx_blocksize)])
        else:
            # default to output_chunk_size
            output_blocks = (args.output_chunk_size,) * image_ndim

        if (args.process_blocksize is not None and
            len(args.process_blocksize) == image_ndim):
            zyx_process_size = args.process_blocksize[::-1] # make it zyx
            process_blocksize = tuple([d if d > 0 else image_shape[di] 
                                        for di,d in enumerate(zyx_process_size)])
        else:
            process_blocksize = output_blocks

        if (args.blocks_overlaps is not None and
            len(args.blocks_overlaps) > 0):
            blocks_overlaps = args.blocks_overlaps[::-1] # make it zyx
        else:
            blocks_overlaps = ()

        if args.eval_channels and len(args.eval_channels) > 0:
            eval_channels = list(args.eval_channels)
        else:
            eval_channels = None

        try:
            logger.info(f'Invoke segmentation with blocksize {process_blocksize}')
            if (args.with_shrink_labels_merge):
                distributed_eval_method = eval_with_shrink_labels_merge
            else:
                distributed_eval_method = eval_with_simple_merge

            # ignore bounding boxes
            output_labels, _ = distributed_eval_method(
                args.input,
                args.input_subpath,
                image_shape,
                args.segmentation_model,
                args.diam_mean,
                process_blocksize,
                args.working_dir,
                dask_client,
                blocksoverlap=blocks_overlaps,
                anisotropy=args.anisotropy,
                min_size=args.min_size,
                resample=(not args.no_resample),
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                stitch_threshold=args.stitch_threshold,
                eval_channels=eval_channels,
                use_torch=args.use_gpu,
                use_gpu=args.use_gpu,
                gpu_device=args.gpu_device,
                max_tasks=args.max_cellpose_tasks,
                iou_threshold=args.iou_threshold,
                iou_depth=args.iou_depth,
                persist_labeled_blocks=args.save_intermediate_labels,
            )

            persisted_labels = write_utils.save(
                output_labels, args.output, output_subpath,
                blocksize=output_blocks,
                resolution=image_attrs.get('pixelResolution'),
                scale_factors=image_attrs.get('downsamplingFactors'),
            )

            if persisted_labels is not None:
                r = dask_client.compute(persisted_labels).result()
                logger.info('DONE!')
            else:
                logger.warning('No segmentation labels were generated')

            dask_client.close()

        except:
            raise


def _print_version_and_exit():
    print(cellpose_version)
    sys.exit(0)


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    try:
        if args.version:
            _print_version_and_exit()

        # prepare logging
        global logger
        logger = configure_logging(args.logging_config, args.verbose)
        logger.info(f'Invoked cellpose segmentation with: {args}')

    except Exception as err:
        print('Logging configuration error:', err)
        traceback.print_exception(err)
        sys.exit(1)

    # run segmentation
    _run_segmentation(args)


if __name__ == '__main__':
    _main()
