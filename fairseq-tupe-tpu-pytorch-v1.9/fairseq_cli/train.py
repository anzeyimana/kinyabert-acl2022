#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import sys
import random
from datetime import datetime

import numpy as np
import torch

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from fairseq import (
    checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

fb_pathmgr_registerd = False


def initialize_loader_for_epoch(args, epoch_itr, prefix='training'):
    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
          fix_batches_to_gpus=False, shuffle=(epoch_itr.epoch >= args.curriculum))
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
          args, itr, epoch_itr.epoch, prefix=prefix, no_progress_bar='simple')
    return progress


def print_model_criterion(model, criterion, args):
      print(model)
      print('| model {}, criterion {}'.format(args.arch,
                                              criterion.__class__.__name__))
      print('| num. model params: {} (num. trained: {})'.format(
          sum(p.numel() for p in model.parameters()),
          sum(p.numel() for p in model.parameters() if p.requires_grad),
      ))


def main(args, init_distributed=False):
    utils.import_user_module(args)

    try:
        from fairseq.fb_pathmgr import fb_pathmgr
        global fb_pathmgr_registerd
        if not fb_pathmgr_registerd:
            fb_pathmgr.register()
            fb_pathmgr_registerd = True
    except (ModuleNotFoundError, ImportError):
        pass

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print_model_criterion(model, criterion, args)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def reset_training_meters(trainer):
    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def reset_perf_training_meters(trainer, i, ignore_index=0):
    if i <= ignore_index:
        trainer.get_meter('wps').reset()
        trainer.get_meter('ups').reset()


def reset_validation_loss_meters(trainer):
    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    progress = initialize_loader_for_epoch(args, epoch_itr)
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second and updates-per-second calculation
        reset_perf_training_meters(trainer, i)

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    reset_training_meters(trainer)


def get_training_stats(trainer, args):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    if getattr(args, 'use_gpu', True):
        # computing perplexity introduces aten::_local_scalar_dense calls
        # that slow training down
        stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    if getattr(args, 'use_gpu', True):
        # computing 'clip' count introduces aten::_local_scalar_dense calls
        # that slow training down, so it's disabled, hence the meter is invalid
        stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        reset_validation_loss_meters(trainer)
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def parse_input_shapes(input_shapes_arg):
    input_shapes = (
        shape.replace('*', 'x').split('x') for shape in input_shapes_arg)
    input_shapes = [list(map(int, shape)) for shape in input_shapes]
    if len(input_shapes) == 1:
        return input_shapes
    input_shapes.sort(key=lambda shape: shape[1])
    errmsg = (
        'Invalid --input_shapes. Batch sizes (dimension 1) need to increase as '
        'num_tokens (dimension 2) decrease. e.g. 16x128 32x64 64x32'
    )
    assert all(
         shape1[0] > shape2[0]
         for shape1, shape2 in zip(input_shapes, input_shapes[1:])), errmsg
    return input_shapes


def now():
    return datetime.now().strftime('%H:%M:%S')


def main_tpu(args):

    def prepare_task(args, xla_device):
        # Setup task, e.g., translation, language modeling, etc.
        task = tasks.setup_task(args)

        # Load valid dataset (we load training data below, based on the latest checkpoint)
        for valid_sub_split in args.valid_subset.split(','):
            task.load_dataset(valid_sub_split, combine=True, epoch=0)

        # Build models and criteria to print some metadata
        torch.manual_seed(args.seed)
        model, criterion = task.build_model(args), task.build_criterion(args)
        xm.master_print(model)
        xm.master_print('| model {}, criterion {}'.format(
            args.arch, criterion.__class__.__name__))
        xm.master_print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)))
        model = model.to(xla_device)
        trainer = Trainer(args, task, model, criterion, xla_device=xla_device)
        lr = trainer.get_lr()

        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator
        # we overwrite distributed args here to shard data using torch_xla's
        # distributed training.
        trainer.args.distributed_rank = xm.get_ordinal()
        trainer.args.distributed_world_size = xm.xrt_world_size()
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
        trainer.args.distributed_rank = 0
        trainer.args.distributed_world_size = 1
        trainer.meters_to_device(xla_device)
        valid_subsets = args.valid_subset.split(',')
        ordinal = xm.get_ordinal(defval=-1)
        device_str = (
            str(xla_device) if ordinal < 0 else
            '{}/{}'.format(xla_device, ordinal)
        )
        return task, trainer, model, epoch_itr, lr, valid_subsets, device_str

    def train_loop_fn(device, trainer, loader, last_batch_index):
        """
        This is the main training loop. It trains for 1 epoch.
        """

        def print_training_update(trainer, progress, args, i):
            stats = get_training_stats(trainer, args=args)
            stats['now'] = now()
            progress.log(stats, tag='train', step=trainer.get_num_updates())
            progress.print_mid_epoch(i+1, force=True)

        stats, log_output, skip_stat_keys = None, None, {'clip'}
        max_update = args.max_update or math.inf
        for i, samples in enumerate(loader, start=epoch_itr.iterations_in_epoch):
            if i == last_batch_index:
                # last batches are incomplete
                break
            log_output = trainer.train_step(samples)
            reset_perf_training_meters(trainer, i, ignore_index=10)
            if (not (i % args.log_steps)) or (i == last_batch_index-1):
                step_args = trainer, progress, args, i
                xm.add_step_closure(print_training_update, args=step_args)
            num_updates = trainer.get_num_updates()
            if (
                not args.disable_validation
                and args.save_interval_updates > 0
                and num_updates % args.save_interval_updates == 0
                and num_updates > 0
            ):
                vloss = validate_subset(
                    args, device, trainer, task, epoch_itr, valid_subsets[0]
                )
                checkpoint_utils.save_checkpoint(
                    args, trainer, epoch_itr, vloss.item(),
                    epoch=epoch, end_of_epoch=False,
                )
            if num_updates >= max_update:
                break


    def valid_loop_fn(
        args, device, trainer, progress, loader, last_batch_index
    ):
        extra_meters = collections.defaultdict(lambda: AverageMeter())
        for i, sample in enumerate(loader):
            if i == last_batch_index:
                # last batches are of different size, will cause recompilations
                break
            log_output = trainer.valid_step(sample)
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)
        stats = get_valid_stats(trainer, args)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        return stats

    def validate_subset(args, device, trainer, task, epoch_itr, subset):
        xm.master_print('Validating the subset "{}", {}'.format(subset, now()))
        # Initialize data iterator
        # we're not sharding the validation set
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on {} \'{}\' subset'.format(device, subset),
            no_progress_bar='simple'
        )
        para_loader = pl.ParallelLoader(progress, [xla_device])
        reset_validation_loss_meters(trainer)
        stats = valid_loop_fn(
            args, device, trainer, progress,
            para_loader.per_device_loader(xla_device), len(progress) - 1
        )
        progress_bar.progress_bar_print(
            progress, stats, step=trainer.get_num_updates(), force=True,
            tag='validate-{}'.format(subset), flush_writer=True,
        )
        xm.master_print('Validated the subset "{}", {}'.format(subset, now()))
        return stats['loss'].avg

    def validate_subsets(args, device, trainer, task, epoch_itr, subsets):
        valid_losses = {
            subset: validate_subset(
                args, device, trainer, task, epoch_itr, subset
            )
            for subset in subsets
        }
        return valid_losses

    def keep_training(lr, epoch_itr, trainer):
        # Train until the learning rate gets too small
        max_epoch = args.max_epoch or math.inf
        max_update = args.max_update or math.inf
        lr, n_updates = trainer.get_lr(), trainer.get_num_updates()
        return ((lr > args.min_lr) and (epoch_itr.epoch < max_epoch) and
            (n_updates < max_update))

    if xu.getenv_as('XLA_USE_BF16', bool, False):
        xm.master_print(
            'WARNING: bfloat16 is enabled. Note that fairseq meters such as '
            'loss will accumulate the numerator, and increment the denominator.'
            ' Due to lack of precision in higher numbers in bfloat16, these '
            'meters will report invalid values after a while.',
            fd=sys.stderr
        )

    xm.master_print('Args', fd=sys.stderr)
    for key, val in args.__dict__.items():
        xm.master_print('\t{} {}'.format(key, val), fd=sys.stderr)
    # `xla_device` is `torch.device` and `device` is `str`
    xla_device = xm.xla_device()
    task, trainer, model, epoch_itr, lr, valid_subsets, device = prepare_task(
        args, xla_device)

    train_meter = StopwatchMeter()
    train_meter.start()
    while keep_training(lr, epoch_itr, trainer):
        # TRAINING
        epoch = epoch_itr.epoch + 1
        xm.master_print('Epoch {} begin {}'.format(epoch, now()))
        progress = initialize_loader_for_epoch(
            args, epoch_itr, prefix='training on {}'.format(device),
        )
        skip_stat_keys = {'clip'}
        if args.suppress_loss_report:
            skip_stat_keys.update({'loss', 'nll_loss', 'gnorm'})
        progress.set_keys_to_skip_mid_epoch(skip_stat_keys)
        para_loader = pl.ParallelLoader(progress, [xla_device])
        train_loop_fn(
            device, trainer, para_loader.per_device_loader(xla_device),
            len(progress) - 1
        )
        training_stats = get_training_stats(trainer, args=args)
        tloss = training_stats['loss'].avg.item()
        progress_bar.progress_bar_print(
            progress, training_stats, tag='train', force=True,
            step=trainer.get_num_updates(), log_xla_metrics=True,
            flush_writer=True,
        )
        xm.master_print('Epoch {} end {}'.format(epoch_itr.epoch, now()))
        if args.metrics_debug:
            xm.master_print(met.metrics_report())
        reset_training_meters(trainer)

        # VALIDATION
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate_subsets(
                args, device, trainer, task, epoch_itr, valid_subsets
            )

            # only use average first validation loss to update learning rate
            vloss = valid_losses[valid_subsets[0]].item()
            xm.master_print('old learning rate: {}'.format(lr))
            lr = trainer.lr_step(epoch_itr.epoch, vloss)
            xm.master_print('new learning rate: {}'.format(lr))
            if args.metrics_debug:
                xm.master_print(met.metrics_report())
        else:
            vloss = None

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(
                args, trainer, epoch_itr, vloss,
                epoch=epoch, end_of_epoch=True,
            )

    train_meter.stop()
    xm.master_print('| done training in {:.1f} seconds'.format(train_meter.sum))
    assert_on_losses(args, train_loss=tloss, valid_loss=vloss)


def assert_on_losses(args, train_loss=None, valid_loss=None):
    if xu.getenv_as('XLA_USE_BF16', bool, False):
        return
    if args.target_valid_loss is not None:
        assert valid_loss is not None and args.target_valid_loss > valid_loss, \
            'valid loss is {}, target is {}'.format(
                valid_loss, args.target_valid_loss
            )
    if args.target_train_loss is not None:
        assert train_loss is not None and args.target_train_loss > train_loss, \
            'train loss is {}, target is {}'.format(
                train_loss, args.target_train_loss
            )


def cli_main_gpu(args):
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


def get_args():
    parser = options.get_training_parser()
    # tpu-comment: need to control certain flags here.
    # e.g. parallelization needs to be suppressed and deferred to torch_xla flags
    # e.g. input tensor shapes need to be controlled via --input_shapes
    parser.add_argument(
        '--input_shapes',
        nargs='*',
        default=None,
        help=(
            'This is used to specify batches and pad lengths. Ex: '
            '`--input_shapes 256x32 512x16` will produce batches w/ 256 '
            'sentences padded to length 32, or 512 sentences padded to length '
            '16. Including too many input shapes will cause graph recompiles and'
            ' degrade performance. On the other extreme, including 1 shape may '
            'waste a ton of flops, since batches may contain a lot of pad '
            'indices on average. Note that the max pad length in this arg will '
            'be used as `--max-source-positions`'))
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--num_cores', type=int, default=8)
    parser.add_argument('--metrics_debug', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--target_train_loss', type=float, default=None)
    parser.add_argument('--target_valid_loss', type=float, default=None)
    parser.add_argument('--suppress_loss_report', action='store_true')
    args = options.parse_args_and_arch(parser)
    return args


def adjust_args_tpu(args):
    if args.fp16:
        raise RuntimeError(
            '--fp16 was provided, this is controlled by env var XLA_USE_BF16')
    print('suppressing distributed_init args for GPU', file=sys.stderr)
    args.distributed_rank = 0
    args.distributed_world_size = 1
    args.distributed_init_method = None
    if args.input_shapes is None:
        raise RuntimeError(
            'Please specify batches and pad lengths using '
            '--input_shapes. Ex: `--input_shapes 256x32 512x16` .'
            'Please refer to the description of the --input_shape'
            ' arg in --help'
        )
    gpu_input_shape_args = ['max_sentences', 'max_sentences_valid', 'max_tokens']
    nonnull_gpu_input_shape_args = [
        arg for arg in gpu_input_shape_args if getattr(args, arg) is not None
    ]
    if nonnull_gpu_input_shape_args:
      errmsg = (
          'On TPUs, please control input shapes '
          'using `--input_shapes`. Any non-null arg in {} will trigger'
          ' this error.'
      ).format(gpu_input_shape_args)
      raise RuntimeError(errmsg)

    args.input_shapes = parse_input_shapes(args.input_shapes)
    args.max_source_positions = args.input_shapes[-1][1]
    # tpu-comment: --log-interval makes progress_bar print after yielding,
    #   thus it's incompatible with torch_xla's data loaders
    if args.log_interval is not None:
        args.log_steps = args.log_steps or args.log_interval
        args.log_interval = None
    return args


def cli_main():
    args = get_args()
    if args.use_gpu:
        return cli_main_gpu(args)
    # From here on out we are in TPU context
    args = adjust_args_tpu(args)
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)


def _mp_fn(index, args):
    torch.set_default_tensor_type('torch.FloatTensor')
    distributed_utils.suppress_output(xm.is_master_ordinal())
    main_tpu(args)


if __name__ == '__main__':
    cli_main()
