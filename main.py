#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

""" TCN for lipreading"""

import pickle
import os
import time
import random
import argparse
import optunity
import optunity.metrics
import numpy as np

import fairseq

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipreading.utils import get_save_folder
from lipreading.utils import load_json, save2npz
from lipreading.utils import load_model, CheckpointSaver
from lipreading.utils import get_logger, update_logger_batch
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.model import Lipreading, get_model_from_json
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.optim_utils import get_optimizer, CosineScheduler
from lipreading.dataloaders import get_data_loaders, get_preprocessing_pipelines

from lipreading.fusion import Fusion

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Pytorch Lipreading ')
    # -- dataset config
    # parser.add_argument('--dataset', default='lrw', help='dataset selection')
    parser.add_argument('--num-classes', type=int, default=500, help='Number of classes')
    parser.add_argument('--modality', default='video', choices=['video', 'raw_audio', 'fusion'], help='choose the modality')
    # -- directory
    parser.add_argument('--data-dir', default='./datasets/LRW_h96w96_mouth_crop_gray', help='Loaded data directory')
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList.txt', help='Path to txt file with labels')
    parser.add_argument('--annonation-direc', default=None, help='Loaded data directory')
    # -- model config
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'], help='Architecture used for backbone')
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'], help='what relu to use' )
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier for mobilenets and shufflenets')
    # -- TCN config
    parser.add_argument('--tcn-kernel-size', type=int, nargs="+", help='Kernel to be used for the TCN module')
    parser.add_argument('--tcn-num-layers', type=int, default=4, help='Number of layers on the TCN module')
    parser.add_argument('--tcn-dropout', type=float, default=0.2, help='Dropout value for the TCN module')
    parser.add_argument('--tcn-dwpw', default=False, action='store_true', help='If True, use the depthwise seperable convolution in TCN architecture')
    parser.add_argument('--tcn-width-mult', type=int, default=1, help='TCN width multiplier')
    # -- train
    parser.add_argument('--training-mode', default='tcn', help='tcn')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--optimizer',type=str, default='adamw', choices = ['adam','sgd','adamw'])
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--init-epoch', default=0, type=int, help='epoch to start at')
    parser.add_argument('--epochs', default=80, type=int, help='number of epochs')
    parser.add_argument('--test', default=False, action='store_true', help='training mode')
    # -- mixup
    parser.add_argument('--alpha', default=0.4, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    # -- test
    parser.add_argument('--model-path', type=str, default=None, help='Pretrained model pathname')
    parser.add_argument('--allow-size-mismatch', default=False, action='store_true',
                        help='If True, allows to init from model with mismatching weight tensors. Useful to init from model with diff. number of classes')
    # -- feature extractor
    parser.add_argument('--extract-feats', default=False, action='store_true', help='Feature extractor')
    parser.add_argument('--mouth-patch-path', type=str, default=None, help='Path to the mouth ROIs, assuming the file is saved as numpy.array')
    parser.add_argument('--mouth-embedding-out-path', type=str, default=None, help='Save mouth embeddings to a specificed path')
    # -- json pathname
    parser.add_argument('--config-path', type=str, default=None, help='Model configuration with json format')
    # -- other vars
    parser.add_argument('--interval', default=50, type=int, help='display interval')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    
    parser.add_argument('--finetune', default=False, action='store_true', help='fusion mode')
    # paths
    parser.add_argument('--logging-dir', type=str, default='./train_logs', help = 'path to the directory in which to save the log file')

    args = parser.parse_args()
    return args


args = load_args()

# torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True

args_loaded = load_json( args.config_path)
tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                    'kernel_size': args_loaded['tcn_kernel_size'],
                    'dropout': args_loaded['tcn_dropout'],
                    'dwpw': args_loaded['tcn_dwpw'],
                    'width_mult': args_loaded['tcn_width_mult'],
                  }

def TSNE_plot(logits, alpha, label_names):
    # PLOT TSNE
    tsne_ = TSNE().fit_transform(logits)
    colors = mcolors.CSS4_COLORS
    cmap = [colors['gold'], colors['orange'], colors['red'], colors['aqua'], colors['deepskyblue'], colors['blue'], colors['indigo'], colors['gray']]
    
    values_by_color = {c: [[], []] for c in cmap}
    for i, v in enumerate(tsne_):
        values_by_color[cmap[int(alpha[i])]][0].append(v[0])
        values_by_color[cmap[int(alpha[i])]][1].append(v[1])
    fig, ax = plt.subplots()
    for k, v in values_by_color.items():
        ax.scatter(v[0], v[1], c=k, label=label_names[cmap.index(k)])

    ax.legend()
    plt.show()
    del tsne_

def extract_feats(model):
    """
    :rtype: FloatTensor
    """
    model.eval()
    preprocessing_func = get_preprocessing_pipelines()['test']
    data = preprocessing_func(np.load(args.mouth_patch_path)['data'])  # data: TxHxW
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])


def evaluate(model, dset_loader, criterion, logger=None, feat_models= None):
    intens_lst = ['0low', '0medium', '0high', '1subtle', '1low', '1medium', '1high', '2none']
    label_names = ['Laughs low', 'Laughs medium', 'Laughs high', 'Smiles subtle', 'Smiles low', 'Smiles medium', 'Smiles high', 'None']
    intensity_count = {x: [] for x in intens_lst}
    
    model.eval()

    running_loss = 0.
    running_corrects = 0.
    with torch.no_grad():
        true_labels = []
        predictions = []
        if type(dset_loader) == dict:
            all_logits = np.zeros((len(list(dset_loader.values())[0])*args.batch_size, args.num_classes))
            all_alpha = np.zeros(len(list(dset_loader.values())[0])*args.batch_size)
            dataset = list(dset_loader.values())[0].dataset
            dset_loader = zip(*dset_loader.values())
        else:
            all_logits = np.zeros((len(dset_loader)*args.batch_size, args.num_classes))
            all_alpha = np.zeros(len(dset_loader)*args.batch_size)
            dataset = dset_loader.dataset

        for batch_idx, ds_ldrs in enumerate(tqdm(dset_loader)):
            if feat_models is not None:
                cat_logits = torch.Tensor().cuda()
                for i, ds in enumerate(ds_ldrs):
                    input, lengths, labels, intensities = ds
                    feat_logits = (feat_models[i](input.unsqueeze(1).cuda(), lengths=lengths))
                    cat_logits = torch.cat((cat_logits, feat_logits), dim=-1)
                logits = model(cat_logits.unsqueeze(1).cuda()).squeeze(1)
            else:
                input, lengths, labels, intensities = ds_ldrs
                logits = model(input.unsqueeze(1).cuda(), lengths=lengths)

            all_logits[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = logits.cpu()
            tmp_alpha = [intens_lst.index(f'{x}{y}') for x, y in zip(labels, intensities)]
            all_alpha[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size] = tmp_alpha

            _, preds = torch.max(F.softmax(logits, dim=-1).data, dim=-1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

            true_labels.extend(labels)
            predictions.extend(preds.cpu())
        
            for i in range(len(intensities)):
                try:
                    intensity_count[f'{labels[i]}{intensities[i]}'].append(int(preds.cpu()[i]))
                except:
                    print(intensities, labels, preds.cpu())

        if args.test:
            TSNE_plot(all_logits, all_alpha, label_names)

        if logger is not None:
            logger.info("{} Confusion Matrix: \n{}".format(args.modality, confusion_matrix(true_labels, predictions, labels=[0, 1, 2])))

            asym_conf_matrix = np.zeros((8, 3), dtype=np.int32)
            for k, c in intensity_count.items():
                for idx in c:
                    asym_conf_matrix[intens_lst.index(k)][idx] += 1

            logger.info(f"{args.modality} Intensity distribution: \n")
            logger.info(f"Preds:\t{'Laughs', 'Smiles', 'None'}")
            for i, k in enumerate(intensity_count.keys()):
                logger.info(f"{k} ({np.sum(asym_conf_matrix[i])} files): \t{asym_conf_matrix[i]*100/np.sum(asym_conf_matrix[i])}")


    print('{} in total\tCR: {}'.format( len(dataset), running_corrects/len(dataset)))
    return running_corrects/len(dataset), running_loss/len(dataset)

def train(model, dset_loader, criterion, epoch, optimizer, logger, feat_models=None, full_finetune=False):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    lr = showLR(optimizer)

    logger.info('-' * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info('Current learning rate: {}'.format(lr))

    if full_finetune and epoch == 0:
        for fmodel in feat_models:
            fmodel.train()

    model.train()
    running_loss = 0.
    running_corrects = 0.
    running_all = 0.
    end = time.time()

    if type(dset_loader) == dict:
        ds_logger = list(dset_loader.values())[0]
        dset_loader = zip(*dset_loader.values())
    else:
        ds_logger = dset_loader

    for batch_idx, ds_ldrs in enumerate(tqdm(dset_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # --
        if feat_models is not None:
            cat_logits = torch.Tensor().cuda()
            for i, ds in enumerate(ds_ldrs):
                input, lengths, labels, _ = ds
                feat_logits = (feat_models[i](input.unsqueeze(1).cuda(), lengths=lengths))
                cat_logits = torch.cat((cat_logits, feat_logits), dim=-1)
            input = cat_logits
            # labels = F.one_hot(labels, num_classes=self.args.num_classes).float().cuda()
        else:
            input, lengths, labels, _ = ds_ldrs

        input, labels_a, labels_b, lam = mixup_data(input, labels, args.alpha)
        labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

        optimizer.zero_grad()
        
        logits = model(input.unsqueeze(1).cuda(), lengths=lengths).view((args.batch_size, args.num_classes)) # output shape = [batch_size, n_classes]

        loss_func = mixup_criterion(labels_a, labels_b, lam)
        loss = loss_func(criterion, logits)

        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # -- compute running performance
        _, predicted = torch.max(F.softmax(logits, dim=-1).data, dim=-1)
        running_loss += loss.item()*input.size(0)
        running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
        running_all += input.size(0)
        # -- log intermediate results
        if batch_idx % args.interval == 0 or (batch_idx == len(ds_logger)-1):
            update_logger_batch( args, logger, ds_logger, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )

    return model



def fusion(args, logger, ckpt_saver, models_path, full_finetune=True):

    fusion_obj = Fusion(args, models_path, logger, tcn_options)

    # -- get model
    model = fusion_obj.load_model(args.model_path)

    # -- get dataset iterators
    dset_loaders, _ = fusion_obj.get_data_loaders(args)

    # -- get loss function
    criterion = nn.CrossEntropyLoss()

    # # -- fix learning rate after loading the ckeckpoint (latency)
    # if args.model_path and args.init_epoch > 0:
    #     scheduler.adjust_lr(optimizer, args.init_epoch-1)

    epoch = args.init_epoch
    while epoch < args.epochs:
        fusion_obj.train() # TODO: adapt train func to match LR_TCN train
        acc_avg_val, loss_avg_val = fusion_obj.evaluate('val')
        logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val, showLR(fusion_obj.optimizer)))
        
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': fusion_obj.fusion_model.state_dict(),
            'optimizer_state_dict': fusion_obj.optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        fusion_obj.scheduler.adjust_lr(fusion_obj.optimizer, epoch)
        epoch += 1

    if full_finetune:
        fusion_obj._load_model_parameters()
        f_epoch = 0
        f_epochs = 20
        while f_epoch < f_epochs:
            fusion_obj.train_full_finetune(f_epoch)
            acc_avg_val, loss_avg_val = fusion_obj.evaluate('val')
            logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', f_epoch, loss_avg_val, acc_avg_val, showLR(fusion_obj.optimizer)))
            
            # -- save checkpoint
            save_dict = {
                'epoch_idx': f_epoch + 1,
                'model_state_dict': fusion_obj.fusion_model.state_dict(),
                'optimizer_state_dict': fusion_obj.optimizer.state_dict()
            }
            ckpt_saver.save(save_dict, acc_avg_val)
            fusion_obj.scheduler.adjust_lr(fusion_obj.optimizer, f_epoch)
            f_epoch += 1
        
    # -- evaluate best-performing epoch on test partition    
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, fusion_obj.fusion_model)
    acc_avg_test, loss_avg_test = fusion_obj.evaluate('test')
    logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))


def main():

    # models_path = {'video': 'train_logs/tcn/fullndc_video_model_realfinetuned/ckpt.best.pth.tar', 'raw_audio': 'train_logs/tcn/fullndc_audio_model_realfinetuned/ckpt.best.pth.tar'}
    # models_path = {'video': '../train_logs/tcn/fullndc_video_model_scratch/ckpt.best.pth.tar', 'raw_audio': '../train_logs/tcn/fullndc_audio_model_scratch/ckpt.best.pth.tar'}
    models_path = {'video': './train_logs/tcn/fullndc_video_model_fullfinetuned/ckpt.best.pth.tar', 'raw_audio': './train_logs/tcn/fullndc_audio_model_lastfinetuned/ckpt.best.pth.tar'}
    # -- logging
    save_path = get_save_folder( args)
    print("Model and log being saved in: {}".format(save_path))
    logger = get_logger(args, save_path)
    ckpt_saver = CheckpointSaver(save_path)

    if args.modality == 'fusion':
        fusion_obj = Fusion(args, models_path, logger, tcn_options)

        model = fusion_obj.fusion_model
        # -- get dataset iterators
        dset_loaders = fusion_obj.get_data_loaders()

        feat_models = fusion_obj.get_feat_models()
    
    else:
        # -- get model
        model = get_model_from_json(args=args)
        # -- get dataset iterators
        dset_loaders, _ = get_data_loaders(args)
        feat_models = None
    
    # -- get loss function
    criterion = nn.CrossEntropyLoss()
    # -- get learning rate scheduler
    scheduler = CosineScheduler(args.lr, args.epochs)
    # -- get optimizer
    optimizer = get_optimizer(args.optimizer, optim_policies=model.parameters(), lr=args.lr)

    if args.model_path:
        assert args.model_path.endswith('.tar') and os.path.isfile(args.model_path), \
            "'.tar' model path does not exist. Path input: {}".format(args.model_path)
        # resume from checkpoint
        if args.modality == 'fusion':
            # -- get model
            model = fusion_obj.load_model(args.model_path)
        else:
            if args.init_epoch > 0:
                model, optimizer, epoch_idx, ckpt_dict = load_model(args.model_path, model, optimizer)
                args.init_epoch = epoch_idx
                ckpt_saver.set_best_from_ckpt(ckpt_dict)
                logger.info('Model and states have been successfully loaded from {}'.format( args.model_path ))
            # init from trained model
            else:
                model = load_model(args.model_path, model, allow_size_mismatch=args.allow_size_mismatch)
                # for param in model.parameters():
                if args.finetune:
                    for param in model.trunk.parameters():
                        param.requires_grad = False
                logger.info('Model has been successfully loaded from {}'.format( args.model_path ))

        # if test-time, performance on test partition and exit. Otherwise, performance on validation and continue (sanity check for reload)
        if args.test:
            acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion, logger=logger, feat_models=feat_models)
            logger.info('Test-time performance on partition {}: Loss: {:.4f}\tAcc:{:.4f}'.format( 'test', loss_avg_test, acc_avg_test))
            return

    # -- fix learning rate after loading the ckeckpoint (latency)
    if args.model_path and args.init_epoch > 0:
        scheduler.adjust_lr(optimizer, args.init_epoch-1)

    epoch = args.init_epoch
    while epoch < args.epochs:
        model = train(model, dset_loaders['train'], criterion, epoch, optimizer, logger, feat_models)
        acc_avg_val, loss_avg_val = evaluate(model, dset_loaders['val'], criterion, logger=logger, feat_models=feat_models)
        logger.info('{} Epoch:\t{:2}\tLoss val: {:.4f}\tAcc val:{:.4f}, LR: {}'.format('val', epoch, loss_avg_val, acc_avg_val, showLR(optimizer)))
        # -- save checkpoint
        save_dict = {
            'epoch_idx': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpt_saver.save(save_dict, acc_avg_val)
        scheduler.adjust_lr(optimizer, epoch)
        epoch += 1

    # -- evaluate best-performing epoch on test partition
    best_fp = os.path.join(ckpt_saver.save_dir, ckpt_saver.best_fn)
    _ = load_model(best_fp, model)
    acc_avg_test, loss_avg_test = evaluate(model, dset_loaders['test'], criterion, logger=logger)
    logger.info('Test time performance of best epoch: {} (loss: {})'.format(acc_avg_test, loss_avg_test))

if __name__ == '__main__':
    main()
