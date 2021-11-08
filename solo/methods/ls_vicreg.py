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
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.linear import Linear
from solo.losses.vicreg import vicreg_loss_func
from solo.methods.base import BaseMethod

@torch.no_grad()
def rotate_tensors(X: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    mask = rot == 1
    X[mask] = torch.flip(X[mask], dims=(3,)).transpose(dim0=2, dim1=3)
    mask = rot == 2
    X[mask] = torch.flip(X[mask], dims=(2,)).flip(dims=(3,))
    mask = rot == 3
    X[mask] = torch.flip(X[mask], dims=(2,)).transpose(dim0=2, dim1=3)
    return X

class LittleSeven_VICReg(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        var_loss_weight: float,
        cov_loss_weight: float,
        **kwargs
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(**kwargs)

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.rot_projector_0_pre = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.rot_projector_1_pre = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.rot_projector_1_aft = nn.Sequential(
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.rot_projector_2_pre = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.rot_projector_2_aft = nn.Sequential(
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.rot_projector_3_pre = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

        self.rot_projector_3_aft = nn.Sequential(
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(LittleSeven_VICReg, LittleSeven_VICReg).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ls_vicreg")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--sim_loss_weight", default=25, type=float)
        parser.add_argument("--var_loss_weight", default=25, type=float)
        parser.add_argument("--cov_loss_weight", default=1.0, type=float)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.rot_projector_0_pre.parameters()},
            {"params": self.rot_projector_1_pre.parameters()},
            {"params": self.rot_projector_2_pre.parameters()},
            {"params": self.rot_projector_3_pre.parameters()},
            {"params": self.rot_projector_1_aft.parameters()},
            {"params": self.rot_projector_2_aft.parameters()},
            {"params": self.rot_projector_3_aft.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the encoder and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        one_labels = torch.LongTensor([1]*batch[1][0].shape[0]).to(batch[1][0].device)
        rot_feats_0, rot_feats_0_2 = out["feats"]
        rotX_1 = rotate_tensors(batch[1][0].clone(),(one_labels*1).detach())
        rotX_2 = rotate_tensors(batch[1][0].clone(),(one_labels*2).detach())
        rotX_3 = rotate_tensors(batch[1][0].clone(),(one_labels*3).detach())

        rot_feats_0 = self.rot_projector_0_pre(rot_feats_0)
        rot_feats_1 = self.rot_projector_1_pre(self.encoder(rotX_1))
        rot_feats_2 = self.rot_projector_2_pre(self.encoder(rotX_2))
        rot_feats_3 = self.rot_projector_3_pre(self.encoder(rotX_3))

        z_1_0_1 = self.rot_projector_1_aft(rot_feats_0)
        z_2_0_2 = self.rot_projector_2_aft(rot_feats_0)
        z_3_0_3 = self.rot_projector_3_aft(rot_feats_0)

        z_1_1_2 = self.rot_projector_1_aft(rot_feats_1)
        z_2_1_3 = self.rot_projector_2_aft(rot_feats_1)
        z_3_1_0 = self.rot_projector_3_aft(rot_feats_1)
        
        z_1_2_3 = self.rot_projector_1_aft(rot_feats_2)
        z_2_2_0 = self.rot_projector_2_aft(rot_feats_2)
        z_3_2_1 = self.rot_projector_3_aft(rot_feats_2)

        z_1_3_0 = self.rot_projector_1_aft(rot_feats_3)
        z_2_3_1 = self.rot_projector_2_aft(rot_feats_3)
        z_3_3_2 = self.rot_projector_3_aft(rot_feats_3)

        vicreg_loss_1_1 = vicreg_loss_func(
            z_1_0_1,
            rot_feats_1.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_2_1 = vicreg_loss_func(
            z_2_3_1,
            rot_feats_1.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_3_1 = vicreg_loss_func(
            z_3_2_1,
            rot_feats_1.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_1 = (vicreg_loss_1_1 + vicreg_loss_2_1 + vicreg_loss_3_1) / 3

        vicreg_loss_1_2 = vicreg_loss_func(
            z_2_0_2,
            rot_feats_2.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_2_2 = vicreg_loss_func(
            z_1_1_2,
            rot_feats_2.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_3_2 = vicreg_loss_func(
            z_3_3_2,
            rot_feats_2.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_2 = (vicreg_loss_1_2 + vicreg_loss_2_2 + vicreg_loss_3_2) / 3

        vicreg_loss_1_3 = vicreg_loss_func(
            z_3_0_3,
            rot_feats_3.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_2_3 = vicreg_loss_func(
            z_2_1_3,
            rot_feats_3.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_3_3 = vicreg_loss_func(
            z_1_2_3,
            rot_feats_3.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_3 = (vicreg_loss_1_3 + vicreg_loss_2_3 + vicreg_loss_3_3) / 3

        vicreg_loss_1_0 = vicreg_loss_func(
            z_3_1_0,
            rot_feats_0.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_2_0 = vicreg_loss_func(
            z_2_2_0,
            rot_feats_0.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_3_0 = vicreg_loss_func(
            z_1_3_0,
            rot_feats_0.detach(),
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        vicreg_loss_0 = (vicreg_loss_1_0 + vicreg_loss_2_0 + vicreg_loss_3_0) / 3

        vicreg_loss = (vicreg_loss_0 + vicreg_loss_1 + vicreg_loss_2 + vicreg_loss_3) / 4

        return class_loss + vicreg_loss
