"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch as th

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)


class DiffusionModel(GaussianDiffusion):
    def __init__(
        self,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__(
            **kwargs,
        )

    def masked_l2(self, a, b):
        bs, n, c = a.shape

        loss = torch.mean(
            torch.norm(
                (a - b).reshape(-1, 6),
                2,
                1,
            )
        )

        return loss

    def training_losses(
        self, model, x_start, t, sparse, model_kwargs=None, noise=None, dataset=None
    ):
    #    self.ddp_model, batch, t, cond, dataset=self.data.dataset,
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start) # torch.Size([4, 32, 64])
        x_t = self.q_sample(x_start, t, noise=noise) # torch.Size([4, 32, 64])

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), sparse, **model_kwargs) # torch.Size([4, 32, 64])

        else:
            raise NotImplementedError(self.loss_type)

        return model_output


    def training_losses_style(
        self, model, x_start, t, sparse, style_code, model_kwargs=None, noise=None, dataset=None
    ):
    #    self.ddp_model, batch, t, cond, dataset=self.data.dataset,
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start) # torch.Size([4, 32, 64])
        x_t = self.q_sample(x_start, t, noise=noise) # torch.Size([4, 32, 64])

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), sparse, style_code.unsqueeze(1).repeat(1,x_t.shape[1],1), **model_kwargs) # torch.Size([4, 32, 64])

        else:
            raise NotImplementedError(self.loss_type)

        return model_output