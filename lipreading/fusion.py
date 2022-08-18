import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import optunity.metrics
import time

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from lipreading.model import MultiscaleMultibranchTCN, get_model_from_json
from lipreading.mixup import mixup_data, mixup_criterion
from lipreading.dataloaders import get_data_loaders
from lipreading.utils import load_model, AverageMeter, showLR, update_logger_batch
from lipreading.optim_utils import get_optimizer, CosineScheduler

class FusionNet(nn.Module):
    def __init__(self, input_size = 1536, num_classes = 3, tcn_options = {}) -> None:
        super(FusionNet, self).__init__()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(input_size, 1024)
        self.fc4 = nn.Linear(1024, num_classes)

        #other layers
        # hidden_dim = 256
        # num_channels = [hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers']
        # self.tcn1 = MultiscaleMultibranchTCN(input_size,
        #     num_channels=num_channels,
        #     num_classes=num_classes,
        #     tcn_options=tcn_options,
        #     dropout=tcn_options['dropout'],
        #     relu_type='prelu',
        #     dwpw=tcn_options['dwpw'])

        # self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, lengths=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

        # B = x.size()[0]
        # return self.tcn_output(self.tcn1(x, lengths, B))

class Fusion():
    def __init__(self, args, models_path, logger=None, tcn_options={}, partitions = ['train', 'val', 'test']) -> None:
        self.args = args
        
        self.feat_models = []
        self.partitions = partitions
        self.modalities = ['video', 'raw_audio']
        self.dset_loaders = {p: {} for p in self.partitions}
        self.data_dirs = dict()
        self.sampler = None
        self.parent_dir= os.path.dirname(args.data_dir)
        self.full_finetune = False
        self.epochs = args.epochs
        self.logger = logger

        self.tcn_options = tcn_options

        self.fusion_model = FusionNet(tcn_options=self.tcn_options).cuda()

        self.init_optimizer = args.optimizer
        self._load_model_parameters()

        for modality in self.modalities:
            self._load_data_dirs(modality)
            self._load_dsets(args, modality)
            if modality == 'video':
                self._load_feats_models(modality, models_path[modality])
            else:
                self._load_feats_models(modality, models_path[modality])
                
                # cp_path = "mnt/c/Users/bohyh/Documents/GITHUB/Thesis/wav2vec_small.pt"
                # model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
                # feature_extractor = model[0].feature_extractor.cuda()
                # self.feat_models.append(feature_extractor)

        for model in self.feat_models:
            model.eval()

    def _load_data_dirs(self, modality):
        feat = 'roi' if modality == 'video' else 'audio'
        self.data_dirs[modality] = os.path.join(*self.parent_dir.split('/'), feat)

    def _load_dsets(self, args, modality):
        ds_ldr, splr = get_data_loaders(args, modality, self.data_dirs[modality], self.sampler)
        for partition in ['train', 'test', 'val']:
            try:
                self.dset_loaders[partition][modality] = ds_ldr[partition]
            except:
                print(f'{partition} not in {args.data_path}')
        if self.sampler is None:
            self.sampler = splr

    def _load_feats_models(self, modality, model_path):
        base_model = get_model_from_json(modality, fusion=True, args=self.args)
        self.feat_models.append(load_model(model_path, base_model, allow_size_mismatch=True).cuda())

    def get_data_loaders(self):
        return self.dset_loaders

    def get_feat_models(self):
        return self.feat_models

    # def _load_model_parameters(self):
    #     self.optimizer = self.init_optimizer
    #     self.lr = self.args.lr
    #     # -- get loss function
    #     self.criterion = nn.CrossEntropyLoss()
    #     # -- get optimizer
    #     self.optimizer = get_optimizer(self.optimizer, optim_policies=self.fusion_model.parameters(), lr=self.lr)
    #     # -- get learning rate scheduler
    #     self.scheduler = CosineScheduler(self.lr, self.epochs)

    # def train(self, model, dset_loader, criterion, epoch, optimizer, logger, feat_models=None, full_finetune=False):
    #     data_time = AverageMeter()
    #     batch_time = AverageMeter()

    #     lr = showLR(optimizer)
        
    #     logger.info('-' * 10)
    #     logger.info('Epoch {}/{}'.format(epoch, self.args.epochs - 1))
    #     logger.info('Current learning rate: {}'.format(lr))

    #     if full_finetune and epoch == 0:
    #         for fmodel in feat_models:
    #             fmodel.train()
        
    #     model.train()
    #     running_loss = 0.
    #     running_corrects = 0.
    #     running_all = 0.
    #     end = time.time()

    #     if type(dset_loader) == dict:
    #         ds_logger = list(dset_loader.values())[0]
    #         dset_loader = zip(*dset_loader.values())
    #     else:
    #         ds_logger = dset_loader

    #     for batch_idx, ds_ldrs in enumerate(tqdm(dset_loader)):
    #         # measure data loading time
    #         data_time.update(time.time() - end)

    #         # --
    #         if feat_models is not None:
    #             cat_logits = torch.Tensor().cuda()
    #             for i, ds in enumerate(ds_ldrs):
    #                 input, lengths, labels, _ = ds
    #                 feat_logits = (feat_models[i](input.unsqueeze(1).cuda(), lengths=lengths))
    #                 cat_logits = torch.cat((cat_logits, feat_logits), dim=-1)
    #             input = cat_logits
    #             # labels = F.one_hot(labels, num_classes=self.args.num_classes).float().cuda()
    #         else:
    #             input, lengths, labels, _ = ds_ldrs

    #         input, labels_a, labels_b, lam = mixup_data(input, labels, self.args.alpha)
    #         labels_a, labels_b = labels_a.cuda(), labels_b.cuda()

    #         optimizer.zero_grad()

    #         logits = model(input.unsqueeze(1).cuda(), lengths=lengths).view((self.args.batch_size, self.args.num_classes)) # output shape = [batch_size, n_classes]

    #         loss_func = mixup_criterion(labels_a, labels_b, lam)
    #         loss = loss_func(self.criterion, logits)

    #         loss.backward()
    #         optimizer.step()
    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #         # -- compute running performance
    #         _, predicted = torch.max(F.softmax(logits, dim=-1).data, dim=-1)
    #         running_loss += loss.item()*input.size(0)
    #         running_corrects += lam * predicted.eq(labels_a.view_as(predicted)).sum().item() + (1 - lam) * predicted.eq(labels_b.view_as(predicted)).sum().item()
    #         running_all += input.size(0)
    #         # -- log intermediate results
    #         if batch_idx % self.args.interval == 0 or (batch_idx == len(ds_logger)-1 ):
    #             update_logger_batch( self.args, logger, ds_logger, batch_idx, running_loss, running_corrects, running_all, batch_time, data_time )
        
    #     return model

    # def evaluate(self, model, dset_loaders, criterion= None, logger=None):
    #     intens_lst = ['0low', '0medium', '0high', '1subtle', '1low', '1medium', '1high', '2none']
    #     intensity_count = {x: [] for x in intens_lst}

    #     model.eval()

    #     running_loss = 0.
    #     running_corrects = 0.
        
    #     with torch.no_grad():
    #         self.true_labels = torch.Tensor().cuda()
    #         self.predictions = torch.Tensor().cuda()
    #         if type(dset_loaders) == dict:
    #             dset_loader = zip(*dset_loaders.values())
    #         for batch_idx, ds_ldrs in enumerate(tqdm(dset_loader)):

    #             if self.args.modality == 'fusion':
    #                 cat_logits = torch.Tensor().cuda()
    #                 for i, ds in enumerate(ds_ldrs):
    #                     input, lengths, labels, intensities = ds
    #                     feat_logits = (self.feat_models[i](input.unsqueeze(1).cuda(), lengths=lengths))
    #                     cat_logits = torch.cat((cat_logits, feat_logits), dim=-1)
    #                 logits = model(cat_logits.cuda())

    #             _, preds = torch.max(F.softmax(logits, dim=-1), dim=-1)
    #             running_corrects += preds.eq(F.one_hot(labels, num_classes=3).float().cuda()).sum().item()
                
    #             loss = criterion(logits.squeeze(1), F.one_hot(labels, num_classes=3).float().cuda())
    #             running_loss += loss.item() * input.size(0)

    #             for i in range(len(intensities)):
    #                 try:
    #                     intensity_count[f'{labels[i]}{intensities[i]}'].append(int(preds.cpu()[i]))
    #                 except:
    #                     print(intensities, labels, preds.cpu())

    #             self.true_labels = torch.cat((self.true_labels, labels.cuda()), dim=-1).type(torch.int8)
    #             self.predictions = torch.cat((self.predictions, preds)).type(torch.int8)
            
            
    #         if self.logger is not None:
    #             self.logger.info("{} Confusion Matrix: \n{}".format(self.args.modality, confusion_matrix(self.true_labels.cpu(), self.predictions.cpu(), labels=[0, 1, 2])))

    #             asym_conf_matrix = np.zeros((8, 3), dtype=np.int32)
    #             for k, c in intensity_count.items():
    #                 for idx in c:
    #                     asym_conf_matrix[intens_lst.index(k)][idx] += 1

    #             self.logger.info(f"{self.args.modality} Intensity distribution: \n")
    #             self.logger.info(f"Preds:\t{'Laughs', 'Smiles', 'None'}")
    #             for i, k in enumerate(intensity_count.keys()):
    #                 self.logger.info(f"{k} ({np.sum(asym_conf_matrix[i])} files): \t{asym_conf_matrix[i]*100/np.sum(asym_conf_matrix[i])}")


    #         print('{} in total\tCR: {}'.format( len(self.dset_loaders['test']['video'].dataset), running_corrects/len(self.dset_loaders['test']['video'].dataset)))

    #     return running_corrects/len(self.dset_loaders['test']['video'].dataset), running_loss/len(self.dset_loaders['test']['video'].dataset)

    def load_model(self, fusion_model_path):
        self.fusion_model = load_model(fusion_model_path, self.fusion_model)
        return self.fusion_model