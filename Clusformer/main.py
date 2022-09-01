# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np

from datasets import ClusterDataset
from models import Clusformer

from utils import dino as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Clusformers', add_help=False)

    # * Model
    parser.add_argument('--num_encoder_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--num_decoder_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=336, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--nheads', default=6, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_querries', default=80, type=int,
                        help="Number of query slots")


    # dataset parameters
    parser.add_argument('--train_feat_path', 
                        default='/home/data/clustering/data/features/part0_train.bin')
    parser.add_argument('--train_label_path', 
                        default='/home/data/clustering/data/labels/part0_train.meta')
    parser.add_argument('--train_knn_graph_path', 
                        default='/home/data/clustering/data/knns/part0_train/faiss_k_80.npz')

    parser.add_argument('--valid_feat_path', 
                        default='/home/data/clustering/data/features/part1_test.bin')
    parser.add_argument('--valid_label_path', 
                        default='/home/data/clustering/data/labels/part1_test.meta')
    parser.add_argument('--valid_knn_graph_path',
                        default='/home/data/clustering/data/knns/part1_test/faiss_k_80.npz')

    parser.add_argument('--feat_dim', default=256, type=int)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")

    # Misc
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=216, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--pred', type=utils.bool_flag, default=False)
    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_dataset = ClusterDataset(
        feat_path=args.train_feat_path,
        label_path=args.train_label_path,
        knn_graph_path=args.train_knn_graph_path,
        feat_dim=args.feat_dim
    )

    valid_dataset = ClusterDataset(
        feat_path=args.valid_feat_path,
        label_path=args.valid_label_path,
        knn_graph_path=args.valid_knn_graph_path,
        feat_dim=args.feat_dim
    )

    
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train data loaded: there are {len(train_dataset)} images.")


    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, shuffle=False)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Valid data loaded: there are {len(valid_dataset)} images.")

    # ============ building Clusformer ... ============

    model = Clusformer(
        num_querries=args.num_querries,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    )

    # move networks to gpu
    model = model.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # ============ preparing loss ... ============
    criterion = nn.BCEWithLogitsLoss()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_data_loader) * args.epochs, eta_min=0)
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        scheduler=scheduler,
        criterion=criterion,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting Clusformer training !")
    for epoch in range(start_epoch, args.epochs):
        train_data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of Clusformer ... ============
        train_stats = train_one_epoch(model, criterion,
            train_data_loader, optimizer, scheduler, epoch, fp16_scaler, args)

        # ============ validate one epoch of Clusformer ... ============
        valid_stats = valid_one_epoch(model, criterion, valid_data_loader, 
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'criterion': criterion.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_train_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        log_valid_stats = {**{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_train_stats) + "\n")
                f.write(json.dumps(log_valid_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # move images to gpu
        # features = [feature.cuda(non_blocking=True) for feature in features]
        features = batch["features"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)
        # print(features.shape)
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logits = model(features)
            loss = criterion(logits, targets)
            f1_score = utils.f1_score(logits, targets) 

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        scheduler.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(f1_score=f1_score.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("[TRAIN] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def valid_one_epoch(model, criterion, data_loader, epoch, fp16_scaler, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # move images to gpu
        # features = [feature.cuda(non_blocking=True) for feature in features]
        features = batch["features"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logits = model(features)
            loss = criterion(logits, targets)
            f1_score = utils.f1_score(logits, targets) 

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(f1_score=f1_score.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("[VALID] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def predict(args):
    valid_dataset = ClusterDataset(
        feat_path=args.valid_feat_path,
        label_path=args.valid_label_path,
        knn_graph_path=args.valid_knn_graph_path,
        feat_dim=args.feat_dim
    )

    # valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, shuffle=False)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        # sampler=valid_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Valid data loaded: there are {len(valid_dataset)} images.")

    # ============ building Clusformer ... ============

    model = Clusformer(
        num_querries=args.num_querries,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    )

    # move networks to gpu
    model = model.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.DataParallel(model)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model
    )

    model.eval()
    predictions = []
    for batch in tqdm(valid_data_loader, total=len(valid_data_loader)):
        features = batch['features'].cuda()
        with torch.no_grad():
            logits = model(features)
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

    predictions = np.concatenate(predictions, axis=0)
    print("Prediction: ", predictions.shape)
    np.save(f"{args.output_dir}/valid_predict.npy", predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clusformer', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.pred:
        predict(args)
    else:
        train(args)
