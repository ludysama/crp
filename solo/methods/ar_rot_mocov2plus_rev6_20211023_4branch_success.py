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
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from solo.losses.ar_rot_moco import moco_loss_func as rot_based_moco_loss_func
from solo.losses.moco import moco_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.metrics import accuracy_at_k
from solo.utils.misc import gather

def rotate_tensors(X: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    mask = rot == 1
    X[mask] = torch.flip(X[mask], dims=(3,)).transpose(dim0=2, dim1=3)
    mask = rot == 2
    X[mask] = torch.flip(X[mask], dims=(2,)).flip(dims=(3,))
    mask = rot == 3
    X[mask] = torch.flip(X[mask], dims=(2,)).transpose(dim0=2, dim1=3)
    return X

# 两个相互独立的旋转标签 rev1
def rotate_batch(batch, branch=2):
    _, X, cls_label = batch
    if branch == 2:
        rot_label_0 = torch.LongTensor([i for i in range(4) for j in range(X[0].shape[0] // 4)]).to(cls_label.device)
        rot_label_1 = torch.LongTensor([i for i in range(4) for j in range(X[0].shape[0] // 4)]).to(cls_label.device)
        X[0] = rotate_tensors(X[0], rot_label_0)
        X[1] = rotate_tensors(X[1], rot_label_1)
        batch = (_, X, cls_label)
        return batch, torch.stack((rot_label_0, rot_label_1))
    if branch == 3:
        rot_label = torch.randint(0, 4, cls_label.shape, device=cls_label.device)
        rot_X = rotate_tensors(X[0].clone().detach(), rot_label)
        return batch, rot_X, rot_label
    if branch == 4:
        rot_label_0 = torch.randint(0, 4, cls_label.shape, device=cls_label.device)
        rot_label_1 = torch.randint(0, 4, cls_label.shape, device=cls_label.device)
        rot_X_0 = rotate_tensors(X[0].clone().detach(), rot_label_0)
        rot_X_1 = rotate_tensors(X[1].clone().detach(), rot_label_1)
        return batch, [rot_X_0, rot_X_1], [rot_label_0, rot_label_1]

class AR_Rotation_MoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor

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
        branch = 1
        self.solo_gar = False
        if not self.solo_gar:
            branch = 2
        non_linear = True
        base_dim = self.features_dim
        # if self.feature_after_moco_projector:
        #     base_dim = proj_output_dim
        if non_linear:
            # a_rot non-linear classifier
            self.a_rot_classifier = nn.Sequential(
                nn.Linear(base_dim*branch, proj_hidden_dim),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, 4**branch)
            )
        else:
            # a_rot linear classifier
            self.a_rot_classifier = nn.Linear(base_dim*branch, 4**branch)

        # relative rotation prediction settings
        self.r_rot_classifier = nn.Sequential(
                nn.Linear(base_dim*2, proj_hidden_dim),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, 4)
            )

        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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
            {"params": self.r_rot_classifier.parameters()}
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
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
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
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

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
        '''
        单分支绝对旋转预测，未加熵约束
        '''
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
        # ar_pred = F.normalize(ar_pred, dim=-1)
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

    def do_dual_grr(self, feats1:torch.Tensor, feats2:torch.Tensor, 
                        feats1_m:torch.Tensor, feats2_m:torch.Tensor ,
                        rot_label1:torch.Tensor, rot_label2:torch.Tensor,use_entropy=False) -> torch.Tensor:
        feats_cat_1 = torch.cat((feats1,feats2_m), dim=1)
        feats_cat_2 = torch.cat((feats2,feats1_m), dim=1)
        rr_pred_1 = self.r_rot_classifier(feats_cat_1)
        rr_pred_2 = self.r_rot_classifier(feats_cat_2)
        if not use_entropy:
            rr_loss = (F.cross_entropy(rr_pred_1, (rot_label1-rot_label2)%4) + F.cross_entropy(rr_pred_2, (rot_label2-rot_label1)%4)) / 2
        else:
            rr_loss_1 = F.cross_entropy(rr_pred_1, (rot_label1-rot_label2)%4, reduction='none')
            rr_loss_2 = F.cross_entropy(rr_pred_2, (rot_label2-rot_label1)%4, reduction='none')
            h_1 = self.calc_entropy(rr_pred_1.detach(), 4)
            h_2 = self.calc_entropy(rr_pred_2.detach(), 4)
            rr_loss = (h_1 * rr_loss_1 + h_2*rr_loss_2).mean(dim=0)
        self.log("dual_grr_loss", rr_loss, on_epoch=True, sync_dist=True)
        return rr_loss

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
        # 将输入数据做旋转
        batch, rot_Xs, rot_labels = rotate_batch(batch, branch=4)

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        total_loss = class_loss

        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        do_rotate = True
        do_moco = True
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

        if do_rotate:
            gar = True
            # tested
            if gar:
                feats_rot_0 = self.encoder(rot_Xs[0])
                feats_rot_1 = self.encoder(rot_Xs[1])
                if self.solo_gar:
                    solo_gar_loss_0, solo_gar_acc1_0 = self.do_solo_gar(feats_rot_0, rot_labels[0], use_entropy=False, use_anti_weights=True)
                    solo_gar_loss_1, solo_gar_acc1_1 = self.do_solo_gar(feats_rot_1, rot_labels[1], use_entropy=False, use_anti_weights=True)
                    solo_gar_loss = (solo_gar_loss_0 + solo_gar_loss_1) / 2
                    solo_gar_acc1 = (solo_gar_acc1_0 + solo_gar_acc1_1) / 2
                    total_loss = total_loss + solo_gar_loss
                    self.log("solo_gar_loss", solo_gar_loss, on_epoch=True, on_step=False, sync_dist=True)
                    self.log("solo_gar_acc1", solo_gar_acc1, on_epoch=True, on_step=False, sync_dist=True)
                else:
                    # dual_gar_loss, dual_gar_acc1 = self.do_dual_gar([feats_rot_0, feats_rot_1], rot_labels, use_entropy=False, use_anti_weights=True)
                    dual_gar_loss, dual_gar_acc1 = self.do_dual_gar([feats_rot_0, feats_rot_1], rot_labels, use_entropy=False, use_anti_weights=False)
                    total_loss = total_loss + 0.3*dual_gar_loss
                    self.log("dual_gar_loss", dual_gar_loss, on_epoch=True, on_step=False, sync_dist=True)
                    self.log("dual_gar_acc1", dual_gar_acc1, on_epoch=True, on_step=False, sync_dist=True)


        if do_moco:
            # ------- contrastive loss -------
            # symmetric
            queue = self.queue.clone().detach()
            nce_loss = (
                moco_loss_func(q1, k2, queue[1], self.temperature)
                + moco_loss_func(q2, k1, queue[0], self.temperature)
            ) / 2

            total_loss = total_loss + nce_loss
            # ------- update queue -------
            keys = torch.stack((gather(k1), gather(k2)))
            self._dequeue_and_enqueue(keys)
            self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return total_loss
