# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from math import degrees
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from solo.losses.ar_rot_moco import moco_loss_func as rot_based_moco_loss_func
from solo.losses.moco import moco_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.metrics import accuracy_at_k
from solo.utils.misc import gather

def check_imgs(X:torch.Tensor) -> None:
    
    class Denormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor

    from matplotlib import pyplot as plt
    import torchvision.transforms as transforms
    import numpy as np
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])
    
    for i in range(len(X)):
        for j in range(4):
            plt.subplot(len(X),4,i*4+j+1)
            fig=plt.imshow(inv_transform(X[i][j].cpu()))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
    plt.show()

@torch.no_grad()
def rotate_tensors(X: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    mask = rot == 1
    X[mask] = torch.flip(X[mask], dims=(3,)).transpose(dim0=2, dim1=3)
    mask = rot == 2
    X[mask] = torch.flip(X[mask], dims=(2,)).flip(dims=(3,))
    mask = rot == 3
    X[mask] = torch.flip(X[mask], dims=(2,)).transpose(dim0=2, dim1=3)
    return X

# 两个相互独立的旋转标签 rev1
@torch.no_grad()
def rotate_batch(batch, branch=4):
    _, X, cls_label = batch
    if branch == 3:
        rot_label = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        mask_r = torch.randperm(cls_label.shape[0])
        mask_m = torch.randperm(cls_label.shape[0])
        rot_label_r = rot_label.clone().detach()[mask_r]
        rot_label_m = rot_label.clone().detach()[mask_m] 
        X[0] = rotate_tensors(X[0], rot_label_m).detach()
        X[1] = rotate_tensors(X[1], rot_label_m).detach()
        rot_X = rotate_tensors(X[2], rot_label_r)
        batch = (_, [X[0], X[1]], cls_label)
        return batch, rot_X, rot_label_r, rot_label_m

    if branch == 4:
        rot_label_r0 = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        rot_label_r1 = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        mask_r0 = torch.randperm(cls_label.shape[0])
        mask_r1 = torch.randperm(cls_label.shape[0])
        rot_label_r0 = rot_label_r0[mask_r0]
        rot_label_r1 = rot_label_r1[mask_r1]
        rot_X0 = rotate_tensors(X[2], rot_label_r0)
        rot_X1 = rotate_tensors(X[3], rot_label_r1)
        # # 下面代码rotation jittering的随机性不是独立的，同一个batch共享一个参数
        with torch.no_grad():
            rot_X0 = transforms.RandomRotation(15)(rot_X0)
            rot_X0 = transforms.CenterCrop(192)(rot_X0)
            rot_X0 = transforms.Resize(224)(rot_X0)
            
            rot_X1 = transforms.RandomRotation(15)(rot_X1)
            rot_X1 = transforms.CenterCrop(192)(rot_X1)
            rot_X1 = transforms.Resize(224)(rot_X1)

        # added some detach
        batch = (_, [X[0].detach(), X[1].detach()], cls_label)
        return batch, [rot_X0.detach(), rot_X1.detach()], rot_label_r0.detach(), rot_label_r1.detach()

    if branch == 6:
        rot_label_r0 = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        rot_label_r1 = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        rot_label_r2 = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        rot_label_r3 = torch.LongTensor([i for i in range(4) for k in range(cls_label.shape[0] // 4)]).to(cls_label.device)
        mask_r0 = torch.randperm(cls_label.shape[0])
        mask_r1 = torch.randperm(cls_label.shape[0])
        mask_r2 = torch.randperm(cls_label.shape[0])
        mask_r3 = torch.randperm(cls_label.shape[0])
        rot_label_r0 = rot_label_r0[mask_r0]
        rot_label_r1 = rot_label_r1[mask_r1]
        rot_label_r2 = rot_label_r0[mask_r2]
        rot_label_r3 = rot_label_r1[mask_r3]
        rot_X0 = rotate_tensors(X[2], rot_label_r0)
        rot_X1 = rotate_tensors(X[3], rot_label_r1)
        rot_X2 = rotate_tensors(X[4], rot_label_r2)
        rot_X3 = rotate_tensors(X[5], rot_label_r3)
        # rot_X0 = transforms.RandomResizedCrop(224, scale=(0.5,1.0),interpolation=transforms.InterpolationMode.BICUBIC)(rot_X0)
        # rot_X1 = transforms.RandomResizedCrop(224, scale=(0.5,1.0),interpolation=transforms.InterpolationMode.BICUBIC)(rot_X1)
        # added some detach
        batch = (_, [X[0].detach(), X[1].detach()], cls_label)
        return batch, [rot_X0.detach(), rot_X1.detach(), rot_X2.detach(), rot_X3.detach], \
            [rot_label_r0.detach(), rot_label_r1.detach(), rot_label_r2.detach(), rot_label_r3.detach()]

class AR_Rotation_MoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor
    label_queue: torch.Tensor
    rev_template: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size
        # self.method_name = 'ar_rot_mocov2plus'

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # absolute rotation prediction settings
        dual_branch = 1
        self.solo_gar = False
        if not self.solo_gar:
            dual_branch = 2
        non_linear = True
        base_dim = self.features_dim
        # if self.feature_after_moco_projector:
        #     base_dim = proj_output_dim
        if non_linear:
            # a_rot non-linear classifier
            self.a_rot_classifier = nn.Sequential(
                nn.Linear(base_dim*dual_branch, proj_hidden_dim),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, 4**dual_branch)
            )
        else:
            # a_rot linear classifier
            self.a_rot_classifier = nn.Linear(base_dim*dual_branch, 4**dual_branch)

        # relative rotation prediction settings
        self.r_rot_classifier = nn.Sequential(
                nn.Linear(base_dim*2, proj_hidden_dim),
                # nn.Dropout(p=0.8),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, 4)
            )

        self.dense_split = 7
        # self.encoder.dense_avgpool = nn.AdaptiveAvgPool2d((self.dense_split, self.dense_split))

        self.dense_feats_dim = 128

        self.a_rev_classifier = nn.Sequential(
                nn.Linear(self.dense_feats_dim*self.dense_split**2, proj_hidden_dim),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, 4)
            )

        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(2, self.queue_size, dtype=torch.long))
        # 注册公转的模板，在开始dense做多少切片确定以后，公转模板也就随之确定，此步骤放在预处理可节省计算时间
        self.register_buffer("rev_template", torch.LongTensor([[i for i in range(self.dense_split**2)] for j in range(4)]))
        # 生成公转模板
        self.generate_rev_template()

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(AR_Rotation_MoCoV2Plus, AR_Rotation_MoCoV2Plus).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.a_rot_classifier.parameters()},
            {"params": self.r_rot_classifier.parameters()},
            {"params": self.a_rev_classifier.parameters()}
            ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, rot_labels=None):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum encoder.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        if rot_labels != None:
            self.label_queue[:, ptr: ptr + batch_size] = rot_labels
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def rotated_loc_code(self, loccode:int, degree:int, split:int, mode:str = 'normal')->int:
        if mode == 'counter':
            degree = (4 - degree) % 4

        for i in range(degree):
            code_row = loccode // split
            code_collumn = loccode % split
            code_row_new = code_collumn
            code_collumn_new = split - code_row -1
            loccode = code_row_new * split + code_collumn_new
            
        return loccode

    @torch.no_grad()
    def generate_rev_template(self):
        for i in range(1,4):
            for j in range(self.dense_split**2):
                self.rev_template[i][j] = self.rotated_loc_code(j,i,self.dense_split)
        self.rev_template = self.rev_template.detach()

    def do_revolution(self, dense: torch.Tensor, target_degree: torch.Tensor, origin_degree: torch.Tensor) -> torch.Tensor:
        res_degree = (target_degree - origin_degree) % 4
        temp_dense = dense.clone()
        for i in range(4):
            temp_dense[res_degree==i] = temp_dense[res_degree==i][:,self.rev_template[i],:]
        return temp_dense

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the online encoder and the online projection.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        q = F.normalize(self.projector(out["feats"]), dim=-1)
        return {**out, "q": q}

    def do_solo_gar(self, feats:torch.Tensor, rot_label:torch.Tensor, use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        ar_pred = self.a_rot_classifier(feats)
        # ar_pred = F.normalize(ar_pred, dim=-1)
        ar_loss = F.cross_entropy(ar_pred, rot_label, reduction='none')
        if use_entropy:
            h = self.calc_entropy(ar_pred.detach(), 4)
            ar_loss = h * ar_loss
        if use_anti_weights:
            anti_weights = 1 - self.current_epoch / self.max_epochs
            ar_loss = anti_weights * ar_loss
        ar_loss = ar_loss.mean(dim=0)
        acc1 = accuracy_at_k(ar_pred, rot_label, top_k=(1,))[0]
        return ar_loss, acc1
    
    def do_dual_gar(self, feats:torch.Tensor, rot_labels:torch.Tensor,use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        feats_cat = torch.cat((feats[0], feats[1]), dim=1)
        ar_pred = self.a_rot_classifier(feats_cat)
        joint_rot_label = rot_labels[0]*4 + rot_labels[1]
        ar_loss = F.cross_entropy(ar_pred, joint_rot_label, reduction='none')
        if use_entropy:
            h = self.calc_entropy(ar_pred.detach(), 16)
            ar_loss = h * ar_loss
        if use_anti_weights:
            anti_weights = 1 - self.current_epoch / self.max_epochs
            ar_loss = anti_weights * ar_loss

        ar_loss = ar_loss.mean(dim=0)
        acc1 = accuracy_at_k(ar_pred, joint_rot_label,top_k=(1,))[0]
        return ar_loss, acc1

    def do_dual_grr(self, feats:torch.Tensor, rot_labels:torch.Tensor,use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        feats_cat = torch.cat((feats[0],feats[1]), dim=1)
        rr_pred = self.r_rot_classifier(feats_cat)
        rr_loss = F.cross_entropy(rr_pred, (rot_labels[1]-rot_labels[0])%4, reduction='none')
        if use_entropy:
            h = self.calc_entropy(rr_pred.detach(), 4)
            rr_loss = h * rr_loss
        if use_anti_weights:
            anti_weights = 1 - self.current_epoch / self.max_epochs
            rr_loss = anti_weights * rr_loss
        rr_loss = rr_loss.mean(dim=0)
        acc1 = accuracy_at_k(rr_pred, (rot_labels[1]-rot_labels[0])%4, top_k=(1,))[0]
        return rr_loss, acc1

    def do_dual_lar(self, denses:torch.Tensor, rot_labels:torch.Tensor, use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        denses[1] = self.do_revolution(denses[1], rot_labels[0], rot_labels[1])

        denses_cat = torch.cat((denses[0],denses[1]), dim=2)
        denses_cat = denses_cat.view(denses_cat.shape[0]*denses_cat.shape[1],-1)

        dense_label1 = torch.stack([rot_labels[0].clone() for j in range(self.dense_split**2)], dim=0).detach()
        dense_label1 = dense_label1.permute(1,0)
        dense_label2 = torch.stack([rot_labels[1].clone() for j in range(self.dense_split**2)], dim=0).detach()
        dense_label2 = dense_label2.permute(1,0)
        dense_label = dense_label1*4 + dense_label2
        dense_label = dense_label.reshape(dense_label.shape[0]*dense_label.shape[1])
        
        denses_ar_pred = self.a_rot_classifier(denses_cat) # or create a new classifier?
        denses_ar_loss = F.cross_entropy(denses_ar_pred, dense_label, reduction='none')
        if use_entropy:
            h = self.calc_entropy(denses_ar_pred.detach(), 16)
            denses_ar_loss = h * denses_ar_loss
        if use_anti_weights:
            anti_weights = 1 - self.current_epoch / self.max_epochs
            denses_ar_loss = anti_weights * denses_ar_loss
        denses_ar_loss = denses_ar_loss.mean(dim=0) / self.dense_split**2
        acc1 = accuracy_at_k(denses_ar_pred, dense_label, top_k=(1,))[0]
        return denses_ar_loss, acc1

    def do_dual_lrr(self, denses:torch.Tensor, rot_labels:torch.Tensor, use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        denses[1] = self.do_revolution(denses[1], rot_labels[0], rot_labels[1])

        denses_cat = torch.cat((denses[0],denses[1]), dim=2)
        denses_cat = denses_cat.view(denses_cat.shape[0]*denses_cat.shape[1],-1)

        dense_label1 = torch.stack([rot_labels[0].clone() for j in range(self.dense_split**2)], dim=0).detach()
        dense_label1 = dense_label1.permute(1,0) 
        dense_label2 = torch.stack([rot_labels[1].clone() for j in range(self.dense_split**2)], dim=0).detach()
        dense_label2 = dense_label2.permute(1,0) 
        dense_label = (dense_label2 - dense_label1) % 4
        dense_label = dense_label.reshape(dense_label.shape[0]*dense_label.shape[1])

        denses_rr_pred = self.r_rot_classifier(denses_cat) # or create a new classifier?
        denses_rr_loss = F.cross_entropy(denses_rr_pred, dense_label, reduction='none')
        if use_entropy:
            h = self.calc_entropy(denses_rr_pred.detach(), 4)
            denses_rr_loss = h * denses_rr_loss
        if use_anti_weights:
            anti_weights = 1 - self.current_epoch / self.max_epochs
            denses_rr_loss = anti_weights * denses_rr_loss
        denses_rr_loss = denses_rr_loss.mean(dim=0) / self.dense_split**2
        acc1 = accuracy_at_k(denses_rr_pred, dense_label, top_k=(1,))[0]
        return denses_rr_loss, acc1

    def do_solo_lrv(self, denses:torch.Tensor, rot_labels:torch.Tensor, use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        """
        使用自身的公转特征0,1,2,3与自身0拼接
        """
        denses_rev1s = []
        denses_rev2s = []
        one_labels = torch.LongTensor([1]*denses[0].shape[0]).to(denses[0].device)

        for i in range(4):
            # 特征的公转增广
            denses_rev1s.append(self.do_revolution(denses[0], (one_labels*i).detach(), (one_labels*0).detach()))
            denses_rev2s.append(self.do_revolution(denses[1], (one_labels*i).detach(), (one_labels*0).detach()))
        
        for i in range(4):
            # 截取部分特征，节约计算量
            denses_rev1s[i] = denses_rev1s[i][:,:,:self.dense_feats_dim].reshape(denses_rev1s[i].shape[0],-1).clone()
            denses_rev2s[i] = denses_rev2s[i][:,:,:self.dense_feats_dim].reshape(denses_rev2s[i].shape[0],-1).clone()
        dense_rev1 = torch.cat(denses_rev1s, dim=0)
        dense_rev2 = torch.cat(denses_rev2s, dim=0)
        # 没做公转增广的数据 公转标签==自身旋转标签
        dense_rev_label1 = torch.cat([((rot_labels[0]+i)% 4).detach() for i in range(4)], dim=0)
        dense_rev_label2 = torch.cat([((rot_labels[1]+i)% 4).detach() for i in range(4)], dim=0)

        denses_rev_pred1 = self.a_rev_classifier(dense_rev1) # or create a new classifier?
        denses_rev_pred2 = self.a_rev_classifier(dense_rev2) # or create a new classifier?
        denses_rev_loss1 = F.cross_entropy(denses_rev_pred1, dense_rev_label1, reduction='none')
        denses_rev_loss2 = F.cross_entropy(denses_rev_pred2, dense_rev_label2, reduction='none')
        if use_entropy:
            h1 = self.calc_entropy(denses_rev_pred1.detach(), 4)
            h2 = self.calc_entropy(denses_rev_pred2.detach(), 4)
            denses_rev_loss1 = h1 * denses_rev_loss1
            denses_rev_loss2 = h2 * denses_rev_loss2
        if use_anti_weights:
            anti_weights = 1 - self.current_epoch / self.max_epochs
            denses_rev_loss1 = anti_weights * denses_rev_loss1
            denses_rev_loss2 = anti_weights * denses_rev_loss2

        # 两个solo分支loss，ACC1取平均
        denses_rev_loss = (denses_rev_loss1 +denses_rev_loss2).mean(dim=0) / 2
        acc1 = (accuracy_at_k(denses_rev_pred1, dense_rev_label1, top_k=(1,))[0] + accuracy_at_k(denses_rev_pred2, dense_rev_label2, top_k=(1,))[0]) /2
        return denses_rev_loss, acc1

    def do_dual_lrv(self, denses:torch.Tensor, rot_labels:torch.Tensor, use_entropy=False, use_anti_weights=False) -> torch.Tensor:
        """
        使用自身的公转特征0,1,2,3与自身0拼接
        逻辑没理清，占个位
        """
        # denses_rev1s = []
        # denses_rev2s = []
        # one_labels = torch.LongTensor([1]*denses[0].shape[0]).to(denses[0].device)

        # for i in range(4):
        #     denses_rev1s.append(self.do_revolution(denses[0], (one_labels*i).detach(), (one_labels*0).detach()))
        #     denses_rev2s.append(self.do_revolution(denses[1], (one_labels*i).detach(), (one_labels*0).detach()))
        
        # for i in range(4):
        #     denses_rev1s[i] = denses_rev1s[i][:,:,:self.dense_feats_dim].reshape(denses_rev1s[i].shape[0],-1).clone()
        #     denses_rev2s[i] = denses_rev2s[i][:,:,:self.dense_feats_dim].reshape(denses_rev2s[i].shape[0],-1).clone()
        # dense_rev1 = torch.cat(denses_rev1s, dim=0)
        # dense_rev2 = torch.cat(denses_rev2s, dim=0)
        # # 默认公转标签是0
        # # dense_rev_label1 = torch.cat([(one_labels*i).detach() for i in range(4)], dim=0)
        # # dense_rev_label2 = torch.cat([(one_labels*i).detach() for i in range(4)], dim=0)
        # # # 默认公转标签是自身旋转标签
        # dense_rev_label1 = torch.cat([((rot_labels[0]+i)% 4).detach() for i in range(4)], dim=0)
        # dense_rev_label2 = torch.cat([((rot_labels[1]+i)% 4).detach() for i in range(4)], dim=0)

        # denses_rev_pred1 = self.a_rev_classifier(dense_rev1) # or create a new classifier?
        # denses_rev_pred2 = self.a_rev_classifier(dense_rev2) # or create a new classifier?
        # denses_rev_loss1 = F.cross_entropy(denses_rev_pred1, dense_rev_label1, reduction='none')
        # denses_rev_loss2 = F.cross_entropy(denses_rev_pred2, dense_rev_label2, reduction='none')
        # if use_entropy:
        #     h1 = self.calc_entropy(denses_rev_pred1.detach(), 4)
        #     h2 = self.calc_entropy(denses_rev_pred2.detach(), 4)
        #     denses_rev_loss1 = h1 * denses_rev_loss1
        #     denses_rev_loss2 = h2 * denses_rev_loss2
        # if use_anti_weights:
        #     anti_weights = 1 - self.current_epoch / self.max_epochs
        #     denses_rev_loss1 = anti_weights * denses_rev_loss1
        #     denses_rev_loss2 = anti_weights * denses_rev_loss2

        # denses_rev_loss = (denses_rev_loss1 +denses_rev_loss2).mean(dim=0) / 2
        # acc1 = (accuracy_at_k(denses_rev_pred1, dense_rev_label1, top_k=(1,))[0] + accuracy_at_k(denses_rev_pred2, dense_rev_label2, top_k=(1,))[0]) /2
        # return denses_rev_loss, acc1


    # 计算预测熵，此处只衡量模型对预测的自信程度，不考虑预测的准确性
    @torch.no_grad()
    def calc_entropy(self, pred:torch.Tensor, cls_dim:int, temperature:float=0.1, scale:float=5.0, low:float=0.01) -> torch.Tensor:
        # parameters warmup
        if self.current_epoch*4 <= self.max_epochs:
            eps = torch.Tensor([self.current_epoch*2/self.max_epochs*3.141592653]).to(pred.device)
            actual_scale = (scale-1)*torch.sin(eps).item()+1
            actual_low = (low-1)*torch.sin(eps).item()+1
        else:
            actual_scale = scale
            actual_low = low

        cls_dim = torch.FloatTensor([cls_dim]).to(pred.device)
        bit_dim = torch.log(cls_dim)
        p = pred / temperature
        p = p.softmax(dim=1)
        h = torch.sum(-p * torch.log(p + 1e-4), dim=1) / bit_dim
        h = (actual_scale-actual_low) * (1-h) + actual_low
        return h

    # 输出global级别和local级别特征的前向传播
    def dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # dense_x = self.encoder.dense_avgpool(x)
        dense_x = torch.flatten(x.clone(),2)
        dense_x = dense_x.permute(0,2,1)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x, dense_x

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """

        # check_imgs([batch[1][0], batch[1][2], batch[1][3]])

        # 将输入数据做旋转
        branch = 4
        if branch == 3:
            batch, rot_X, rot_label_r0, rot_label_r1 = rotate_batch(batch, branch=3)
        if branch == 4:
            batch, rot_Xs, rot_label_r0, rot_label_r1 = rotate_batch(batch, branch=4)
        if branch == 6:
            batch, rot_Xs, rot_labels = rotate_batch(batch, branch=6)

        # 检查输入数据，使用检查代码时需要在MobaXterm里跑
        # print(rot_label_r0[:4], rot_label_r1[:4])
        # check_imgs([batch[1][0], batch[1][1], rot_Xs[0], rot_Xs[1]])
        
        # 欺骗moco的基本模型，使得可以继承solo-learn作者写的主框架
        self.num_crops = 2
        
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        total_loss = class_loss

        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        do_rotate = True
        do_moco = False

        if do_rotate:
            get_locoal_features = False
            feats_rot0 , dense_rot0, feats_rot1, dense_rot1 = None, None, None, None

            do_global_rotation = True
            do_local_rotation = False
            do_local_revolution = False
            assert not (get_locoal_features and (do_local_revolution or do_local_rotation)), "local tasks can't be done without local features"

            gr_loss, gar_loss, grr_loss = None, None, None
            lr_loss, lar_loss, lrr_loss = None, None, None
            lrv_loss = None
            rotation_loss = None

            if branch == 3: 
                if get_locoal_features:
                    feats_rot0, dense_rot0 = self.dense_forward(rot_X)
                    feats_rot0, dense_rot1 = self.dense_forward(batch[1][0])
                else:
                    feats_rot0 = self.encoder(rot_X)
                    feats_rot1 = feats1.clone()
            elif branch == 4:
                if get_locoal_features:
                    feats_rot0, dense_rot0 = self.dense_forward(rot_Xs[0])
                    feats_rot1, dense_rot1 = self.dense_forward(rot_Xs[1])
                else:
                    feats_rot0 = self.encoder(rot_Xs[0])
                    feats_rot1 = self.encoder(rot_Xs[1])
            elif branch == 6:
                rot_label_r0 = rot_labels[2]
                rot_label_r1 = rot_labels[3]
                if get_locoal_features:
                    feats_rot0, dense_rot0 = self.dense_forward(rot_Xs[2])
                    feats_rot1, dense_rot1 = self.dense_forward(rot_Xs[3])
                else:
                    feats_rot0 = self.encoder(rot_Xs[2])
                    feats_rot1 = self.encoder(rot_Xs[3])

            if do_global_rotation:
                # 结合moco框架下不要用solo_gar分支
                if self.solo_gar:
                    assert False, "don't use solo_gar in current version"
                    # solo_gar_loss_0, solo_gar_acc1_0 = self.do_solo_gar(feats_rot0, rot_label_r, use_entropy=False, use_anti_weights=False)
                    # solo_gar_loss_1, solo_gar_acc1_1 = self.do_solo_gar(feats_rot1, rot_label_m, use_entropy=False, use_anti_weights=False)
                    # solo_gar_loss = (solo_gar_loss_0 + solo_gar_loss_1) / 2
                    # solo_gar_acc1 = (solo_gar_acc1_0 + solo_gar_acc1_1) / 2
                    # gar_loss = solo_gar_loss
                    # self.log("solo_gar_loss", solo_gar_loss, on_epoch=True, on_step=False, sync_dist=True)
                    # self.log("solo_gar_acc1", solo_gar_acc1, on_epoch=True, on_step=False, sync_dist=True)
                else:
                    dual_gar_loss, dual_gar_acc1 = self.do_dual_gar([feats_rot0.clone(), feats_rot1.clone()], [rot_label_r0.clone(), rot_label_r1.clone()], use_entropy=False, use_anti_weights=False)
                    '''
                        注释？
                    '''
                    gar_loss = dual_gar_loss
                    self.log("dual_gar_loss", dual_gar_loss, on_epoch=True, on_step=False, sync_dist=True)
                    self.log("dual_gar_acc1", dual_gar_acc1, on_epoch=True, on_step=False, sync_dist=True)
                
                dual_grr_loss, dual_grr_acc1 = self.do_dual_grr([feats_rot0.clone(), feats_rot1.clone()], [rot_label_r0.clone(), rot_label_r1.clone()], use_entropy=False, use_anti_weights=False)    
                self.log("dual_grr_loss", dual_grr_loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log("dual_grr_acc1", dual_grr_acc1, on_epoch=True, on_step=False, sync_dist=True)
                grr_loss = dual_grr_loss
                if self.current_epoch < 1 and False: 
                    gr_loss = 0.6*gar_loss
                else:
                    # gr_loss = 0.4*gar_loss + 0.1*grr_loss
                    # gr_loss = 0*gar_loss + 0.04*grr_loss
                    gr_loss = 1*gar_loss + 0*grr_loss

            if do_local_rotation:
                dual_lar_loss, dual_lar_acc1 = self.do_dual_lar([dense_rot0, dense_rot1], [rot_label_r0, rot_label_r1], use_entropy=True, use_anti_weights=False)
                self.log("dual_lar_loss", dual_lar_loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log("dual_lar_acc1", dual_lar_acc1, on_epoch=True, on_step=False, sync_dist=True)
                lar_loss = dual_lar_loss

                dual_lrr_loss, dual_lrr_acc1 = self.do_dual_lrr([dense_rot0, dense_rot1], [rot_label_r0, rot_label_r1], use_entropy=False, use_anti_weights=False)
                self.log("dual_lrr_loss", dual_lrr_loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log("dual_lrr_acc1", dual_lrr_acc1, on_epoch=True, on_step=False, sync_dist=True)
                lrr_loss = dual_lrr_loss

                lr_loss = 0.4*lar_loss + 0.1*lrr_loss
                # lr_loss = lar_loss
                # lr_loss = 0.1*lrr_loss

            if do_local_revolution:
                dual_larv_loss, dual_larv_acc1 = self.do_solo_lrv([dense_rot0, dense_rot1], [rot_label_r0, rot_label_r1], use_entropy=False, use_anti_weights=False)
                self.log("dual_lrv_loss", dual_larv_loss, on_epoch=True, on_step=False, sync_dist=True)
                self.log("dual_lrv_acc1", dual_larv_acc1, on_epoch=True, on_step=False, sync_dist=True)
                lrv_loss = dual_larv_loss

            if gr_loss != None:
                if rotation_loss == None:
                    rotation_loss = gr_loss
                else:
                    rotation_loss += gr_loss
            if lr_loss != None:
                if rotation_loss == None:
                    rotation_loss = lr_loss
                else:
                    rotation_loss += lr_loss
            if lrv_loss != None:
                if rotation_loss == None:
                    rotation_loss = lrv_loss
                else:
                    rotation_loss += lrv_loss

            # moco默认学习率0.3，但是旋转任务在学习率0.15左右才开始收敛，所以这里加个0.5权重
            total_loss = total_loss + rotation_loss

        if do_moco:
            q1 = self.projector(feats1)
            q2 = self.projector(feats2)
            q1 = F.normalize(q1, dim=-1)
            q2 = F.normalize(q2, dim=-1)

            with torch.no_grad():
                k1 = self.momentum_projector(momentum_feats1)
                k2 = self.momentum_projector(momentum_feats2)
                k1 = F.normalize(k1, dim=-1)
                k2 = F.normalize(k2, dim=-1)
            # ------- contrastive loss -------
            # symmetric
            queue = self.queue.clone().detach()
            
            # 3分支下，moco分支带有旋转特征，可以参加正反例划分
            if branch == 3: 
                label_queue = self.label_queue.clone().detach()
                nce_loss = (rot_based_moco_loss_func(q1, k2, queue[1], label_queue[1], rot_label_r1, self.temperature) + 
                            rot_based_moco_loss_func(q2, k1, queue[0], label_queue[0], rot_label_r1, self.temperature)) / 2
                queue_rot_labels = torch.stack((gather(rot_label_r1), gather(rot_label_r1)), dim=0)
                # ------- update queue -------
                keys = torch.stack((gather(k1), gather(k2)))
                self._dequeue_and_enqueue(keys, rot_labels=queue_rot_labels)

            # 4分支下，moco分支中不引入旋转特征，使用原版的moco_loss即可
            elif branch == 4:
                nce_loss = (
                    moco_loss_func(q1, k2, queue[1], self.temperature)
                    + moco_loss_func(q2, k1, queue[0], self.temperature)
                ) / 2
                # ------- update queue -------
                keys = torch.stack((gather(k1), gather(k2)))
                self._dequeue_and_enqueue(keys)

            # 解决一开始loss为nan的临时措施
            if self.current_epoch > 1 and False:
                total_loss = total_loss + nce_loss
            else:
                total_loss = total_loss + 1*nce_loss

            self.log("train_nce_loss", nce_loss, on_epoch=True, on_step=False, sync_dist=True)

        return total_loss
