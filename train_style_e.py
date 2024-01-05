# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import json
import os
import random
import numpy as np
import torch
from data_loaders.style_e_dataset import StyleEDataset
from model.networks import PureMLP
from runner.train_mlp import train_step
from runner.training_loop import TrainLoop, TrainLoop_clip
from torch.utils.data import DataLoader
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args
import clip

from torch.utils.tensorboard import SummaryWriter
from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from collections import OrderedDict
import glob
from models.audio_encoder import *

def load_model(args):
    print("creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]

    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    model, diffusion = create_model_and_diffusion(args)

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev())
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    return model, diffusion

def train_diffusion_model(args, dataloader, test_dataloader, config,start_epoch,currBestLoss, facemodel, model, diffusion, audioencoder,writer, text_encoder):

    print("Training...")
    TrainLoop_clip(args, model, diffusion, dataloader, test_dataloader, config,facemodel, audioencoder,start_epoch,currBestLoss,writer, text_encoder).run_loop()
    print("Done.")


def train_mlp_model(args, dataloader):
    print("creating MLP model...")
    args.arch = args.arch[len("mlp_") :]
    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    model = PureMLP(
        args.latent_dim,
        args.input_motion_length,
        args.layers,
        args.sparse_dim,
        args.motion_nfeat,
    )
    model.train()

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev())
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0

    while (nb_iter + 1) < args.num_steps:

        for (motion_target, motion_input) in dataloader:

            loss, optimizer, current_lr = train_step(
                motion_input,
                motion_target,
                model,
                optimizer,
                nb_iter,
                args.num_steps,
                args.lr,
                args.lr / 10.0,
                dist_util.dev(),
                args.lr_anneal_steps,
            )
            avg_loss += loss
            avg_lr += current_lr

            if (nb_iter + 1) % args.log_interval == 0:
                avg_loss = avg_loss / args.log_interval
                avg_lr = avg_lr / args.log_interval

                print("Iter {} Summary: ".format(nb_iter + 1))
                print(f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) == args.num_steps:
                break
            nb_iter += 1

        with open(
            os.path.join(args.save_dir, "model-iter-" + str(nb_iter + 1) + ".pth"),
            "wb",
        ) as f:
            torch.save(model.state_dict(), f)



def multi2single(pretrain):
    new_state_dict = OrderedDict()
    for key, value in pretrain.items():
        name = key[7:]
        new_state_dict[name] = value
    return new_state_dict


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(unexpected_keys) != 0:
        state_dict_new = {}
        for key in state_dict.keys():
            state_dict_new[key.replace("module.", "")] = state_dict[key]
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict_new, strict=False
        )
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])

def main():
    args = train_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.overwrite = True
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    args.batch_size  =  128
    args.motion_nfeat  =  64
    args.sparse_dim  =  64
    args.input_motion_length  =  32
    args.latent_dim  =  512
    args.style_code_dim = 512
    args.config  =  "config/Style-E/style_e.json"
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)

    dataset = StyleEDataset(root_dir = 'processed_MEAD_front', is_train=True)
    test_dataset = StyleEDataset(root_dir = 'processed_MEAD_front',is_train=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)


    start_epoch = 0
    currBestLoss = 1e5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder, preprocess = clip.load("ViT-B/32", device=device)
    text_encoder.eval()
    requires_grad(text_encoder, False)

    facemodel = ParametricFaceModel(
        bfm_folder='Deep3DFaceRecon_pytorch/BFM', camera_distance=10.0, focal=1015.0, center=112.0,
        is_train=False, default_name='BFM_model_front.mat'
    )
    audioencoder = AudioEncoder()
    audioencoder_ckpt = torch.load(args.audioencoder_ckpt)
    audioencoder.load_state_dict(audioencoder_ckpt)
    audioencoder = audioencoder.to(device)


    print("creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]

    num_gpus = torch.cuda.device_count()
    args.num_workers = args.num_workers * num_gpus

    model, diffusion = load_model(args)

    model.train()
    num_gpus = 0

    if num_gpus > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dist_util.setup_dist()
        model = torch.nn.DataParallel(model).cuda()
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.module.parameters()) / 1000000.0)
        )
    else:
        dist_util.setup_dist(args.device)
        model.to(dist_util.dev())
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )

    tag = config['tag']
    pipeline = config['pipeline']
    writer = SummaryWriter('runs/debug_{}_{}'.format(tag, pipeline))

    train_diffusion_model(args, dataloader, test_dataloader, config,start_epoch,currBestLoss,facemodel, model, diffusion, audioencoder, writer, text_encoder)


if __name__ == "__main__":
    main()
