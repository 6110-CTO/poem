"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent and EATA code.
"""
from logging import debug
import os
import time
import argparse
import json
import random
import numpy as np
import uuid
from pathlib import Path
from math import sqrt
from pycm import *
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.utils.data import ConcatDataset
import math
import pandas as pd
import json
from typing import ValuesView
from loguru import logger
# from utils.utils import get_logger

from dataset.selectedRotateImageFolder import prepare_test_data
from utils.cli_utils import *

import torch    
import torch.nn.functional as F

import tent
import eata
import sar
from cotta.imagenet import cotta
import poem
from sam import SAM
import timm
import protector as protect
from temperature_scaling import ModelWithTemperature, _ECELoss, TemperatureModel
import models.Res as Resnet


def run(data_loader, model, args):
    ents = []
    accs1 = []
    accs5 = []
    logits_list = []
    labels_list = []

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, dl in enumerate(data_loader):

            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            output = model(images).detach()

            # _, targets = output.max(1)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            accs1.extend(acc1.tolist())
            accs5.extend(acc5.tolist())
            logits_list.append(output)
            labels_list.append(target)

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            ents.extend(softmax_ent(output).tolist())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # break
            if i % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break

    with torch.no_grad():
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        ece_criterion = _ECELoss().cuda()
        ece = ece_criterion(logits, labels.view(-1)).item()

        model_delta = get_models_delta(model, get_model(args).to(args.device))

    info = {
        'top1': top1.avg.item(),
        'top5': top5.avg.item(),
        'accs1': accs1,
        'accs5': accs5,
        'ents': ents,
        'ece':ece,
        'model_delta': model_delta,
    }

    return info



def get_args():

    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--data', default='/datasets/ImageNet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/datasets/ImageNet/ImageNet-C', help='path to corruption dataset')
    parser.add_argument('--v2_path', default='/datasets/ImageNet2/imagenetv2-matched-frequency-format-val/', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=1, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='eata', type=str, help='no_adapt, tent, eata, sar, cotta, poem')
    parser.add_argument('--model', default='resnet50_gn_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='normal', type=str, help='normal, continual, bs1, in_dist, natural_shift, severity_shift, eps_cdf, martingale')
    parser.add_argument('--cont_size', default=5000, type=int, help='each corruption size for continual type')
    parser.add_argument('--severity_list', nargs="+", type=int, default=[5, 4, 3, 2, 1, 2, 3, 4, 5])
    parser.add_argument('--temp', type=float, default=1, help='temperature for the model to be calibrated')
    parser.add_argument('--exp_comment', type=str, default='')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;')

    # PEM parameters
    parser.add_argument('--gamma', type=float, help='protector\'s gamma', default=1 / (8 * sqrt(3)))
    parser.add_argument('--eps_clip', type=float, help='clipping value for epsilon during protection', default=1.80)
    parser.add_argument('--lr_factor', type=float, default=1, help='multiplies the learning rate for poem')
    parser.add_argument('--vanilla_loss', action='store_false', dest='vanilla_loss', help='Use vanilla match loss (not l match ++).')

    return parser.parse_args()



def get_model(args):
    # build model for adaptation
    bs = args.test_batch_size
    if args.method in ['tent', 'eata', 'sar', 'cotta','no_adapt', 'poem']:
        if args.model == "resnet50_gn_timm":
            net = timm.create_model('resnet50_gn', pretrained=True)
            args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            args.temp = 0.90 # SAR transofrms
            # args.temp = 0.875 # timm transforms
        elif args.model == "vitbase_timm":
            net = timm.create_model('vit_base_patch16_224', pretrained=True)
            args.lr = (0.001 / 64) * bs
            args.temp = 1.025 # SAR transforms
            # args.temp = 0.875 # timm transforms
        elif args.model == "resnet50_bn_torch":
            net = Resnet.__dict__['resnet50'](pretrained=True)
            # init = torch.load("./pretrained_models/resnet50-19c8e357.pth")
            # net.load_state_dict(init)
            args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
        else:
            assert False, NotImplementedError
        net = net.cuda()
    else:
        assert False, NotImplementedError
    
    net = ModelWithTemperature(net, args.temp)
    return net


if __name__ == '__main__':

    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.output): # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)

    run_infos = []
    args.adapt = True
    args.timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    args.exp_name = args.timestamp + "--{}-level{}-seed{}".format(args.model, args.level, args.seed) + "_" + str(uuid.uuid4())[:6]
    output_path = Path(args.output)  / 'imagenet' / args.method / args.exp_type / args.exp_name
    output_path.parent.mkdir(exist_ok=True, parents=True)
    logger.add(f"{output_path}.log")
    
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    if args.exp_type == 'eps_cdf':
        common_corruptions = ['brightness']
    if args.exp_type == 'martingale':
        common_corruptions = ['brightness']
        common_corruptions = [item for item in common_corruptions for _ in range(2)]

    if args.exp_type == 'ablation':
        args.test_batch_size = 1


    if args.exp_type == 'bs1':
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")


    dummy_model_for_transorms = get_model(args)


    if args.exp_type == 'continual':
        datasets = []
        sev = args.level
        for i, cpt in enumerate(common_corruptions):
            logger.info(f'adding {cpt} to continual data with level {sev}')
            args.corruption = cpt
            val_dataset, _, holdout_dataset, holdout_loader = prepare_test_data(args, model=dummy_model_for_transorms.model)
            indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(args.seed + i)).tolist()[:args.cont_size]
            val_dataset = Subset(val_dataset, indices)
            datasets.append(val_dataset)

        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of continual dataset is {len(mixed_dataset)}")
        logger.info(f"not shuffling dataset")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        common_corruptions = [f'continual_{sev}']

    if args.exp_type == 'severity_shift':
        datasets = []
        corruption = 'gaussian_noise'
        args.corruption = corruption
        for i, sev in enumerate(args.severity_list):
            logger.info(f'adding {corruption} with level {sev} to severity shift data')
            args.level = sev
            val_dataset, _, holdout_dataset, holdout_loader = prepare_test_data(args, model=dummy_model_for_transorms.model)
            indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(args.seed + i)).tolist()[:args.cont_size]
            val_dataset = Subset(val_dataset, indices)
            datasets.append(val_dataset)

        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of continual dataset is {len(mixed_dataset)}")
        logger.info(f"not shuffling dataset")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        common_corruptions = [f'severity_shift_{corruption}']


    if args.exp_type == 'natural_shift':
        common_corruptions = ['v2']

    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")
    
    if args.exp_type == 'in_dist':
      args.corruption = 'original'
      common_corruptions = ['original']
      val_dataset, val_loader, holdout_dataset, holdout_loader = prepare_test_data(args, model=dummy_model_for_transorms.model)
      
    acc1s, acc5s = [], []
    ir = args.imbalance_ratio

    for i, corrupt in enumerate(common_corruptions):
        args.corruption = corrupt
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs
        
        if args.method in ['tent', 'eata', 'sar', 'no_adapt', 'poem', 'cotta']:
            if args.exp_type not in ['mix_shifts', 'continual', 'in_dist', 'severity_shift']:
                val_dataset, val_loader, holdout_dataset, holdout_loader = prepare_test_data(args, model=dummy_model_for_transorms.model)
        else:
            assert False, NotImplementedError

        net = get_model(args)

        if args.test_batch_size == 1 and args.method == 'sar':
            args.lr = 2 * args.lr
            logger.info("double lr for sar under bs=1")

        logger.info(args)

        if args.method == "tent":
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            tented_model = tent.Tent(net, optimizer)

            run_info = run(val_loader, tented_model, args)

        elif args.method == "no_adapt":
            tented_model = net
            run_info = run(val_loader, tented_model, args)

        elif args.method == "eata":

            num_samples = min(args.fisher_size, len(holdout_dataset))
            indices = torch.randperm(len(holdout_dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()[:num_samples]

            logger.info('prepping fisher dataset + matrix')
            fisher_dataset = Subset(holdout_dataset, indices)
            fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True)

            net = eata.configure_model(net)
            params, param_names = eata.collect_params(net)
            # fishers = None
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    targets = targets.cuda(args.gpu, non_blocking=True)
                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)

            run_info = run(val_loader, adapt_model, args)

        elif args.method in ['sar']:
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            logger.info(param_names)

            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)

            run_info = run(val_loader, adapt_model, args)

        elif args.method == "poem":
            # on holdout for protector
            net = get_model(args)
            info_on_holdout = run(holdout_loader, net, args)
            info_on_holdout['method'] = 'holdout'
            info_on_holdout.update(**vars(args))

            # resetting the model
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)

            if args.exp_type == 'martingale':
                args.adapt = i % 2 == 0

            protector = protect.get_protector_from_ents(info_on_holdout['ents'], args)
            optimizer = torch.optim.SGD(params, args.lr * args.lr_factor, momentum=0.9)
            adapt_model = poem.POEM(net, optimizer, protector, e0=args.sar_margin_e0, adapt=args.adapt, vanilla_loss=args.vanilla_loss)
            run_info = run(val_loader, adapt_model, args)
            run_info['epsilons'] = adapt_model.protector.epsilons
            run_info['u_before'] = adapt_model.protector.info['u_before']
            run_info['u_after'] = adapt_model.protector.info['u_after']
            run_info['martingales'] = adapt_model.protector.martingales
        elif args.method == "cotta":
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            # optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)

            if args.model == "resnet50_gn_timm":
                args.lr = (1e-3 / 32)
                # args.temp = 0.875 # timm transforms
            elif args.model == "vitbase_timm":
                args.lr = (1e-3 / 16)

            logger.info(f"COTTA LR {args.lr}")
            optimizer = torch.optim.Adam(params, args.lr, betas=(0.9, 0.999), weight_decay=0.0)

            adapt_model = cotta.CoTTA(net, optimizer)
            run_info = run(val_loader, adapt_model, args)
        else:
            assert False, NotImplementedError

        run_info.update(**vars(args))

        for k, v in run_info.items():
            if isinstance(v, np.ndarray):
                run_info[k] = v.tolist()
            if isinstance(v, torch.Tensor):
                run_info[k] = v.detach().cpu().tolist()

        run_infos.append(run_info)
        logger.info(f"Top1 Acc     - {run_info['top1']:.4f}")
        logger.info(f"Top5 Acc     - {run_info['top5']:.4f}")
        logger.info(f"ECE          - {run_info['ece']:.4f}")
        logger.info(f"Model Delta  - {run_info['model_delta']:.4f}")

df = pd.DataFrame(run_infos)
df = df.drop(columns=['accs1', 'accs5', 'ents'])
try:
    df = df.drop(columns=['epsilons'])
except:
    pass

df.to_csv(f"{output_path}.csv", index=False)

with open(f"{str(output_path)}.json", "w") as outfile:
    json.dump(run_infos, outfile)
