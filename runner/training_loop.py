# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/openai/guided-diffusion
# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import functools

import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import clip
from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import create_named_schedule_sampler, LossAwareSampler
from torch.optim import AdamW
from tqdm import tqdm
from utils import dist_util

def SSIM_loss(seq1, seq2, size_average=True):
    mu1 = seq1.mean(dim=1, keepdim=True)
    mu2 = seq2.mean(dim=1, keepdim=True)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = seq1.var(dim=1, unbiased=False, keepdim=True)
    sigma2_sq = seq2.var(dim=1, unbiased=False, keepdim=True)
    sigma12 = ((seq1 * seq2).mean(dim=1, keepdim=True)) - (mu1_mu2)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.mean(1).mean(1)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



class TrainLoop_clip:
    def __init__(self, args, model, diffusion, data, test_dataloader, config, facemodel, audioencoder,start_epoch,currBestLoss,writer, text_encoder):
        self.args = args
        self.config = config
        self.data_eval = test_dataloader
        self.text_encoder = text_encoder
        self.audioencoder = audioencoder

        self.facemodel = facemodel
        self.start_epoch = start_epoch
        self.currBestLoss = currBestLoss
        self.writer = writer

        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.load_optimizer = args.load_optimizer
        self.use_fp16 = False
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.l1_criterion = nn.L1Loss()
        self.l2_criterion = nn.MSELoss()
        self.cross_criterion = nn.CrossEntropyLoss()

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step and self.load_optimizer:
            self._load_optimizer_state()

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint,
                    map_location=dist_util.dev(),
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )

        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        assert os.path.exists(opt_checkpoint), "optimiser states does not exist."
        state_dict = dist_util.load_state_dict(
            opt_checkpoint, map_location=dist_util.dev()
        )
        self.opt.load_state_dict(state_dict)

    def train(self, epoch):
        totalSteps = len(self.data)


        avgLoss = 0
        count = 0

        print(f"Starting epoch {epoch}")
        for bii, bi in enumerate(self.data):
            loss_total = self.run_step(bi, epoch)
            avgLoss += loss_total.item()

            count += 1

            if bii % self.config['log_step'] == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'\
                        .format(epoch, self.config['num_epochs'], bii, totalSteps,
                                avgLoss / count))



            self.step += 1
        self.writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)


    def eval(self, epoch):
        totalSteps = len(self.data_eval)

        avgLoss = 0
        count = 0

        print(f"Starting epoch {epoch}")
        for bii, bi in enumerate(self.data_eval):
            loss_total = self.run_step_eval(bi, epoch)
            avgLoss += loss_total.item()
            count += 1
            
            if bii % self.config['log_step'] == 0:
                print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'\
                        .format(epoch, self.config['num_epochs'], bii, totalSteps,
                                avgLoss / count))



            self.step += 1
        testLoss = avgLoss / totalSteps
        self.writer.add_scalar('Loss/val_totalLoss', avgLoss / totalSteps, epoch)

        if testLoss < self.currBestLoss:
            print('>>>> saving best epoch {}'.format(epoch), testLoss)
            self.save()
        else:
            if epoch%self.config['save_epoch'] == 0:
                self.save()
    def run_loop(self):
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train(epoch)
            self.eval(epoch)

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def calu_mouth_loss(self, shape_a, shape_b, bfm):
        lip_points = [51,57,62,66]
        lip_points_id = bfm.keypoints[lip_points]

        lip_diff_a = shape_a[:, lip_points_id[ ::2]] - shape_a[:, lip_points_id[1::2]]
        lip_diff_a = torch.sum(lip_diff_a**2, dim=2)

        lip_diff_b = shape_b[:, lip_points_id[ ::2]] - shape_b[:, lip_points_id[1::2]]
        lip_diff_b = torch.sum(lip_diff_b**2, dim=2)

        return F.l1_loss(lip_diff_a, lip_diff_b)

    def generate_stylecode_clip(self, au_sentence):
        text = clip.tokenize(au_sentence).to(self.device)
        text_features = self.text_encoder.encode_text(text)

        return text_features
    def gen(self, parameters, style_parameters,style_au_sentence, t):

        target_style_code_clip = self.generate_stylecode_clip(style_au_sentence)

        generate = functools.partial(
            self.diffusion.training_losses_style,
            self.ddp_model,
            style_parameters[:,:,80:144],
            t,
            parameters[:,:,80:144],
            target_style_code_clip,
            dataset=self.data.dataset,
        )

        results = generate() 

        recon_loss = 0
        mouth_loss = 0
        cur_shapes = []
        for i in range(0,32):
            tmp = style_parameters[:,i].clone().detach()
            tmp[:,80:144] = results[:,i]

            style_pred_dict = self.facemodel.split_coeff(tmp)
            style_pred_shape = self.facemodel.compute_shape(style_pred_dict['id'], style_pred_dict['exp'])
            
            cur = style_parameters[:,i].clone().detach()
            cur_dict = self.facemodel.split_coeff(cur)
            cur_shape = self.facemodel.compute_shape(cur_dict['id'], cur_dict['exp'])
            cur_shapes.append(cur_shape)


            mouth_loss += self.calu_mouth_loss(style_pred_shape, cur_shape, self.facemodel)

            tmp = parameters[:,i].clone().detach()

        recon_loss += mouth_loss/32
        recon_loss += self.l2_criterion(results, style_parameters[:,:,80:144])

        loss = recon_loss

        return loss


    def run_step(self, bi, epoch):
        loss_total = self.forward_backward(bi, epoch)
        self.mp_trainer.optimize(self.opt)

        self._step_lr()
        return loss_total


    def run_step_eval(self, bi, epoch):
        loss_total = self.forward_eval(bi, epoch)

        return loss_total

    def forward_eval(self, bi, epoch):
        self.mp_trainer.zero_grad()
        parameters, style_au_sentence, style_mel_feature, style_parameters = bi



        parameters = parameters.type(torch.FloatTensor).cuda()
        style_mel_feature = style_mel_feature.type(torch.FloatTensor).cuda()
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()


        t, weights = self.schedule_sampler.sample(style_parameters.shape[0], dist_util.dev())

        with torch.no_grad():
            loss = self.gen(parameters, style_parameters,style_au_sentence, epoch, t)



        return loss.detach()

    def forward_backward(self, bi, epoch):
        self.mp_trainer.zero_grad()
        parameters, style_au_sentence, style_mel_feature, style_parameters = bi

        parameters = parameters.type(torch.FloatTensor).cuda()
        style_mel_feature = style_mel_feature.type(torch.FloatTensor).cuda()
        style_parameters = style_parameters.type(torch.FloatTensor).cuda()


        t, weights = self.schedule_sampler.sample(style_parameters.shape[0], dist_util.dev())

        loss = self.gen(parameters, style_parameters,style_au_sentence, epoch, t)


        self.mp_trainer.backward(loss)
        return loss.detach()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _step_lr(self):
        # One-step learning rate decay if needed.
        if not self.lr_anneal_steps:
            return
        if (self.step + self.resume_step) > self.lr_anneal_steps:
            self.lr = self.lr / 30.0
            self.lr_anneal_steps = False
        else:
            self.lr = self.lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self, epoch=0):
        return f"model_"+str(epoch)+f"_{(self.step+self.resume_step):09d}.pt"

    def save(self, epoch = 0):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log("saving model...")
            filename = self.ckpt_file_name(epoch)

            if not os.path.exists(self.config["model_path"]):
                os.makedirs(self.config["model_path"])
            with open(
                os.path.join(self.config["model_path"], filename),
                "wb",
            ) as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with open(
            os.path.join(self.config["model_path"], f"opt_"+str(epoch)+f"_{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
