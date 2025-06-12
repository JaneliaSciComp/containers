import os
import sys
import traceback

import io_utils.read_utils as read_utils
import io_utils.write_utils as write_utils

from dask.distributed import (Client, LocalCluster)

from distributed_cellpose.impl_v1 import (distributed_eval as eval_with_labels_dt_merge)
from distributed_cellpose.impl_v2 import (distributed_eval as eval_with_iou_merge)
from distributed_cellpose.preprocessing import get_preprocessing_steps

from io_utils.zarr_utils import prepare_attrs

from utils.configure_logging import (configure_logging)
from utils.configure_dask import (load_dask_config, ConfigureWorkerPlugin)


def _floattuple(arg):
    if arg is not None and arg.strip():
        return tuple([float(d) for d in arg.split(',')])
    else:
        return ()


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _stringlist(arg):
    if arg is not None and arg.strip():
        return list(filter(lambda x: x, [s.strip() for s in arg.split(',')]))
    else:
        return []

def _intlist(arg):
    if arg is not None and arg.strip():
        return [int(d) for d in arg.split(',')]
    else:
        return []

def _define_args():
    from cellpose.cli import get_arg_parser

    args_parser = get_arg_parser()
    args_parser.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             help = "input directory")
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "input subpath")
    args_parser.add_argument('--timeindex',
                             dest='input_timeindex',
                             type=int,
                             default=0,
                             help = "input time index")
    args_parser.add_argument('--input-channels',
                             dest='input_channels',
                             type=_intlist,
                             help = "input segmentation channels")

    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=_floattuple,
                             help = "voxel spacing")

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
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')

    args_parser.add_argument('--use-model-only-to-eval', '--use_model_only_to_eval',
                             dest='eval_only_with_model',
                             action='store_true',
                             default=False,
                             help='If true it uses only CellposeModel to eval otherwise it uses both CellposeModel and SizeModel')

    args_parser.add_argument('--test-mode', '--test_mode',
                             dest='test_mode',
                             action='store_true',
                             default=False,
                             help='Test-mode')

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
    distributed_args.add_argument('--device', required=False, default='0', type=str,
                                  dest='device',
                                  help='which device to use, use an integer for torch, or mps for M1')    
    distributed_args.add_argument('--models-dir', dest='models_dir',
                                  type=str,
                                  help='cache cellpose models directory')
    distributed_args.add_argument('--model',
                                  dest='segmentation_model',
                                  type=str,
                                  default='cyto3',
                                  help='A builtin segmentation model or a model added to the cellpose models directory')
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
    distributed_args.add_argument('--label-distance-threshold', '--label-dist-th',
                                  dest='label_dist_th',
                                  type=float,
                                  default=1.0,
                                  help='Label distance transform threshold used for merging labels'),
    distributed_args.add_argument('--save-intermediate-labels',
                                  action='store_true',
                                  dest='save_intermediate_labels',
                                  default=False,
                                  help='Save intermediate labels as zarr')
    distributed_args.add_argument('--merge-labels-iou-only',
                                  action='store_true',
                                  dest='merge_labels_with_iou',
                                  default=False,
                                  help='Only use IOU to merge labels')
    
    distributed_args.add_argument('--preprocessing-steps', '--preprocessing_steps',
                                  dest='preprocessing_steps',
                                  type=_stringlist,
                                  default=[],
                                  help='Preprocessing steps')

    distributed_args.add_argument('--preprocessing-config', '--preprocessing_config',
                                  dest='preprocessing_config',
                                  type=str,
                                  help='Preprocessing steps parameters')
    
    distributed_args.add_argument('--logging-config', dest='logging_config',
                                  type=str,
                                  help='Logging configuration')

    return args_parser


def _run_segmentation(args):
    load_dask_config(args.dask_config)
    if args.models_dir is not None:
        models_dir = os.path.realpath(args.models_dir)
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = models_dir
    elif os.environ.get('CELLPOSE_LOCAL_MODELS_PATH'):
        models_dir = os.environ['CELLPOSE_LOCAL_MODELS_PATH']
    else:
        models_dir = None

    if models_dir:
        from cellpose.models import get_user_models
        logger.info(f'Download cellpose models to {models_dir}')
        get_user_models()

    if args.dask_scheduler:
        dask_client = Client(address=args.dask_scheduler)
        logger.info(f'Initialize Dask Worker plugin with: {models_dir}, {args.logging_config}')
    else:
        # use a local asynchronous client
        dask_client = Client(LocalCluster(silence_logs=not args.verbose))

    worker_config = ConfigureWorkerPlugin(models_dir,
                                          args.logging_config,
                                          args.verbose,
                                          worker_cpus=args.worker_cpus)
    dask_client.register_plugin(worker_config, name='WorkerConfig')

    image_data, image_attrs = read_utils.open(args.input, args.input_subpath,
                                              data_timeindex=args.input_timeindex,
                                              data_channels=args.input_channels)
    image_ndim = image_data.ndim
    image_shape = image_data.shape
    image_dtype = image_data.dtype
    image_data = None


    if args.voxel_spacing:
        voxel_spacing = read_utils.get_voxel_spacing({}, args.voxel_spacing)
    else:
        voxel_spacing = read_utils.get_voxel_spacing(image_attrs, (1.,) * image_ndim)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            expansion = args.expansion_factor
        else:
            expansion = 1.
        voxel_spacing /= expansion

    logger.info(f'Image data shape/dim/dtype: {image_shape}, {image_ndim}, {image_dtype}')
    
    if args.output:
        output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

        if args.process_blocksize is not None:
            if len(args.process_blocksize) < image_ndim:
                # append 0s
                process_blocksize_arg = (args.process_blocksize +
                                        (0,) * (image_ndim - len(args.process_blocksize)))
            else:
                process_blocksize_arg = args.process_blocksize
            zyx_process_size = process_blocksize_arg[::-1] # make it zyx
            process_blocksize = tuple([d if d > 0 else image_shape[di]
                                        for di,d in enumerate(zyx_process_size)])
        else:
            process_blocksize = image_shape # process the whole image

        if (args.blocks_overlaps is not None and
            len(args.blocks_overlaps) > 0):
            blocks_overlaps = args.blocks_overlaps[::-1] # make it zyx
        else:
            blocks_overlaps = ()

        try:
            logger.info(f'Invoke segmentation with blocksize {process_blocksize}')
            if (args.merge_labels_with_iou):
                distributed_eval_method = eval_with_iou_merge
            else:
                distributed_eval_method = eval_with_labels_dt_merge

            if args.anisotropy and args.anisotropy != 1.0:
                anisotropy = args.anisotropy
            else:
                if voxel_spacing is not None:
                    anisotropy = voxel_spacing[0] / voxel_spacing[1]
                else:
                    anisotropy = None

            preprocessing_steps = get_preprocessing_steps(args.preprocessing_steps, 
                                                          args.preprocessing_config,
                                                          voxel_spacing=voxel_spacing)
            logger.info(f'Preprocessing steps: {preprocessing_steps}')

            if args.z_axis is not None and args.z_axis >= 0:
                z_axis = args.z_axis
            else:
                z_axis = 0 # default to first axis
            if args.channel_axis is not None and args.channel_axis >= 0:
                channel_axis = args.channel_axis
            else:
                channel_axis = None
            # ignore bounding boxes
            output_labels, _ = distributed_eval_method(
                args.input,
                args.input_subpath,
                image_shape,
                args.input_timeindex,
                args.input_channels,
                args.segmentation_model,
                process_blocksize,
                args.working_dir,
                dask_client,
                blocksoverlap=blocks_overlaps,
                diameter=args.diam_mean,
                min_size=args.min_size,
                anisotropy=anisotropy,
                z_axis=z_axis,
                channel_axis=channel_axis,
                normalize=not args.no_norm,
                normalize_percentile=args.norm_percentile,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                stitch_threshold=args.stitch_threshold,
                flow3D_smooth=args.flow3D_smooth,
                iou_depth=args.iou_depth,
                iou_threshold=args.iou_threshold,
                label_dist_th=args.label_dist_th,
                persist_labeled_blocks=args.save_intermediate_labels,
                preprocessing_steps=preprocessing_steps,
                test_mode=args.test_mode,
                use_gpu=args.use_gpu,
                gpu_device=args.gpu_device,
            )

            labels_attributes = prepare_attrs(output_subpath,
                                              axes=image_attrs.get('axes'),
                                              coordinateTransformations=image_attrs.get('coordinateTransformations'),
                                              pixelResolution=image_attrs.get('pixelResolution'),
                                              downsamplingFactors=image_attrs.get('downsamplingFactors'))

            if args.output_blocksize is not None:
                if len(args.output_blocksize) < output_labels.ndim:
                    # append 0s
                    output_blocksize_arg = (args.output_blocksize +
                                            (0,) * (output_labels.ndim - len(args.output_blocksize)))
                else:
                    output_blocksize_arg = args.output_blocksize
                zyx_blocksize = output_blocksize_arg[::-1] # make it zyx
                output_blocks = tuple([d if d > 0 else output_labels.shape[di]
                                    for di,d in enumerate(zyx_blocksize)])
            else:
                # default to output_chunk_size
                output_blocks = (args.output_chunk_size,) * output_labels.ndim

            persisted_labels = write_utils.save(
                output_labels, args.output, output_subpath,
                blocksize=output_blocks,
                container_attributes=labels_attributes,
            )

            if persisted_labels is not None:
                r = dask_client.compute(persisted_labels).result()
                logger.info(f'DONE ({r})!')
            else:
                logger.warning('No segmentation labels were generated')

            dask_client.close()

        except:
            raise


def _print_version_and_exit():
    from cellpose import version_str as cellpose_version

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
