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
from solo.losses.ar_rot_moco import moco_loss_func
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
    # rot_label_0 = torch.randint(0, 4, cls_label.shape, device=cls_label.device)
    # rot_label_1 = torch.randint(0, 4, cls_label.shape, device=cls_label.device)
    if branch == 2:
        rot_label_0 = torch.LongTensor([i for i in range(4) for j in range(X[0].shape[0] // 4)]).to(cls_label.device)
        rot_label_1 = torch.LongTensor([i for i in range(4) for j in range(X[0].shape[0] // 4)]).to(cls_label.device)
        X[0] = rotate_tensors(X[0], rot_label_0)
        X[1] = rotate_tensors(X[1], rot_label_1)
        batch = (_, X, cls_label)
        return batch, torch.stack((rot_label_0, rot_label_1))
    if branch == 3:
        rot_label = torch.randint(0, 4, cls_label.shape, device=cls_label.device)
        rot_X = rotate_tensors(X[0], rot_label)
        return batch, rot_X, rot_label

class AR_Rotation_MoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor
    queue_rot_index: torch.Tensor
    rot_template: torch.Tensor

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

        # 用来做moco与rotpred特征的解耦
        self.moco_rot_decouple = False
        if self.moco_rot_decouple:
            non_linear_decouple = False
            if non_linear_decouple:
                self.decoupling_mlp = nn.Sequential(
                    nn.Linear(self.features_dim, proj_hidden_dim),
                    nn.BatchNorm1d(proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(proj_hidden_dim,self.features_dim*2)
                )
                self.momentum_decoupling_mlp = nn.Sequential(
                    nn.Linear(self.features_dim, proj_hidden_dim),
                    nn.BatchNorm1d(proj_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(proj_hidden_dim,self.features_dim*2)
                )
            else:
                self.decoupling_mlp = nn.Sequential(
                    nn.Linear(self.features_dim, self.features_dim*2),
                    nn.BatchNorm1d(self.features_dim*2)
                )
                self.momentum_decoupling_mlp = nn.Sequential(
                    nn.Linear(self.features_dim, self.features_dim*2),
                    nn.BatchNorm1d(self.features_dim*2)
                )


        add_projector_layer = False
        if add_projector_layer:
            # projector
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )

            # momentum projector
            self.momentum_projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        else:
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

        self.feature_after_resnet = True
        self.feature_after_moco_projector = False
        assert not (self.feature_after_moco_projector and self.feature_after_resnet), 'only one of the global absolute rotation pridictoin can be selected'

        # absolute rotation prediction settings
        branch = 1
        non_linear = True
        base_dim = self.features_dim
        if self.feature_after_moco_projector:
            base_dim = proj_output_dim
        if non_linear:
            # a_rot non-linear classifier
            self.a_rot_classifier = nn.Sequential(
                nn.Linear(base_dim*branch, proj_hidden_dim),
                nn.ReLU(),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.Linear(proj_hidden_dim, 4**branch)
            )
        else:
            # a_rot linear classifier
            self.a_rot_classifier = nn.Linear(base_dim*branch, 4**branch)

        # relative rotation prediction settings
        self.r_rot_classifier = nn.Sequential(
                nn.Linear(base_dim*2, proj_hidden_dim),
                nn.ReLU(),
                # nn.BatchNorm1d(proj_hidden_dim),
                nn.Linear(proj_hidden_dim, 4)
            )


        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.register_buffer("queue_rot_index", torch.zeros(2, queue_size, dtype=torch.long))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        # self.queue_rot_index = nn.functional.normalize(self.queue_rot_index, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("rot_template", torch.zeros(queue_size, dtype=torch.long))
        self.rot_template = torch.LongTensor([i for k in range(queue_size//self.batch_size) for i in range(4) for j in range(self.batch_size//4)])
        # self.rot_template = torch.stack([self.rot_template]*self.batch_size)

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
        if self.moco_rot_decouple:
            extra_learnable_params.append({"params": self.decoupling_mlp.parameters()})
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        if self.moco_rot_decouple:
            extra_momentum_pairs.append((self.decoupling_mlp, self.momentum_decoupling_mlp))
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

    def do_solo_gar(self, feats1:torch.Tensor, feats2:torch.Tensor, 
                        feats1_m:torch.Tensor, feats2_m:torch.Tensor ,
                        rot_label1:torch.Tensor, rot_label2:torch.Tensor,use_entropy=False) -> torch.Tensor:
        '''
        单分支绝对旋转预测，未加熵约束
        '''
        ar_pred_1 = self.a_rot_classifier(feats1)
        ar_pred_2 = self.a_rot_classifier(feats2)
        if not use_entropy:
            ar_loss = (F.cross_entropy(ar_pred_1,rot_label1) + F.cross_entropy(ar_pred_2,rot_label2)) / 2
        else:
            ar_loss_1 = F.cross_entropy(ar_pred_1,rot_label1, reduction='none')
            ar_loss_2 = F.cross_entropy(ar_pred_2,rot_label2, reduction='none')
            h_1 = self.calc_entropy(ar_pred_1.detach(), 4)
            h_2 = self.calc_entropy(ar_pred_2.detach(), 4)
            ar_loss = (h_1 * ar_loss_1 + h_2*ar_loss_2).mean(dim=0)

        acc1 = (accuracy_at_k(ar_pred_1, rot_label1, top_k=(1,))[0] + accuracy_at_k(ar_pred_2, rot_label2, top_k=(1,))[0]) / 2 
        self.log("solo_gar_loss", ar_loss, on_epoch=True, sync_dist=True)
        self.log("solo_gar_acc1", acc1, on_epoch=True, on_step=False, sync_dist=True)
        return ar_loss
    
    def do_dual_gar(self, feats1:torch.Tensor, feats2:torch.Tensor, 
                        feats1_m:torch.Tensor, feats2_m:torch.Tensor ,
                        rot_label1:torch.Tensor, rot_label2:torch.Tensor,use_entropy=False) -> torch.Tensor:
        feats_cat_1 = torch.cat((feats1,feats2_m), dim=1)
        feats_cat_2 = torch.cat((feats2,feats1_m), dim=1)
        ar_pred_1 = self.a_rot_classifier(feats_cat_1)
        ar_pred_2 = self.a_rot_classifier(feats_cat_2)
        if not use_entropy:
            ar_loss = (F.cross_entropy(ar_pred_1, rot_label1*4+rot_label2) + F.cross_entropy(ar_pred_2, rot_label2*4+rot_label1)) / 2
        else:
            ar_loss_1 = F.cross_entropy(ar_pred_1, rot_label1*4+rot_label2, reduction='none')
            ar_loss_2 = F.cross_entropy(ar_pred_2, rot_label2*4+rot_label1, reduction='none')
            h_1 = self.calc_entropy(ar_pred_1.detach(), 16)
            h_2 = self.calc_entropy(ar_pred_2.detach(), 16)
            ar_loss = (h_1 * ar_loss_1 + h_2*ar_loss_2).mean(dim=0)
        self.log("dual_gar_loss", ar_loss, on_epoch=True, sync_dist=True)
        return ar_loss

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
        if self.current_epoch*4 <= self.max_epochs and False:
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
        branch = 2
        if branch == 2:
            batch, rot_labels= rotate_batch(batch, branch=branch)
        elif branch == 3: # 这段代码不能在这个版本中使用
            batch, rot_X, rot_label = rotate_batch(batch, branch=branch)

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        total_loss = class_loss

        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        # 如果做特征解耦就在这里加个mlp，否则只是改个名字
        if self.moco_rot_decouple:
            feats1 = self.decoupling_mlp(feats1)
            feats2 = self.decoupling_mlp(feats2)
            feats1_moco, feats1_rot = torch.split(feats1, self.features_dim, dim=1)
            # feats1_moco = F.normalize(feats1_moco,dim=-1)
            # feats1_rot = F.normalize(feats1_rot,dim=-1)
            feats2_moco, feats2_rot = torch.split(feats2, self.features_dim, dim=1)
            # feats2_moco = F.normalize(feats2_moco,dim=-1)
            # feats2_rot = F.normalize(feats2_rot,dim=-1)
            with torch.no_grad():
                momentum_feats1 = self.momentum_decoupling_mlp(momentum_feats1)
                momentum_feats2 = self.momentum_decoupling_mlp(momentum_feats2)
                momentum_feats1_moco, momentum_feats1_rot = torch.split(momentum_feats1,self.features_dim, dim=1)
                # momentum_feats1_moco = F.normalize(momentum_feats1_moco, dim=-1)
                # momentum_feats1_rot = F.normalize(momentum_feats1_rot, dim=-1)
                momentum_feats2_moco, momentum_feats2_rot = torch.split(momentum_feats2,self.features_dim, dim=1)
                # momentum_feats2_moco = F.normalize(momentum_feats2_moco, dim=-1)
                # momentum_feats2_rot = F.normalize(momentum_feats2_rot, dim=-1)
        else:
            feats1_moco = feats1
            feats1_rot = feats1
            feats2_moco = feats2
            feats2_rot = feats2
            momentum_feats1_moco = momentum_feats1
            momentum_feats1_rot = momentum_feats1
            momentum_feats2_moco = momentum_feats2
            momentum_feats2_rot = momentum_feats2   

        do_rotate = True
        do_moco = True
        if do_moco:
            q1 = self.projector(feats1_moco)
            q2 = self.projector(feats2_moco)
            q1 = F.normalize(q1, dim=-1)
            q2 = F.normalize(q2, dim=-1)

            with torch.no_grad():
                k1 = self.momentum_projector(momentum_feats1_moco)
                k2 = self.momentum_projector(momentum_feats2_moco)
                k1 = F.normalize(k1, dim=-1)
                k2 = F.normalize(k2, dim=-1)

        if do_rotate:
            gar = True
            dual_gar_after_resnet = False
            dual_grr_after_resnet = False
            # tested
            if gar:
                if self.feature_after_resnet:
                    solo_gar_loss = self.do_solo_gar(feats1_rot, feats2_rot, momentum_feats1_rot, momentum_feats2_rot, rot_labels[0], rot_labels[1], use_entropy=False)
                    total_loss = total_loss + solo_gar_loss
                if self.feature_after_moco_projector:
                    solo_gar_loss = self.do_solo_gar(q1, q2, k1, k2, rot_labels[0], rot_labels[1], use_entropy=False)
                    total_loss = total_loss + solo_gar_loss
            # not tested
            if dual_gar_after_resnet:
                dual_gar_loss = self.do_dual_gar(feats1_rot, feats2_rot, momentum_feats1_rot, momentum_feats2_rot, rot_labels[0], rot_labels[1], use_entropy=False)
                total_loss = total_loss + dual_gar_loss
            # not tested    
            if dual_grr_after_resnet:
                dual_grr_loss = self.do_dual_grr(feats1_rot, feats2_rot, momentum_feats1_rot, momentum_feats2_rot, rot_labels[0], rot_labels[1], use_entropy=False)
                total_loss = total_loss + dual_grr_loss

        if do_moco:
            # ------- contrastive loss -------
            # symmetric
            queue = self.queue.clone().detach()
            nce_loss = (
                moco_loss_func(q1, k2, queue[1], self.rot_template, self.temperature)
                + moco_loss_func(q2, k1, queue[0], self.rot_template, self.temperature)
            ) / 2
            total_loss = total_loss + nce_loss
            # ------- update queue -------
            keys = torch.stack((gather(k1), gather(k2)))
            self._dequeue_and_enqueue(keys)
            self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return total_loss
