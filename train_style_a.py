import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import argparse
import math
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from util import *
from data_loaders.config import Config
from data_loaders.style_a_dataset import StyleADATASET
from torch.utils.data import DataLoader
from model.stylegan import lpips
from model.stylegan.model import Generator, Downsample
from model.vtoonify import VToonify, ConditionalDiscriminator, Audio_Driven_VToonify
from model.bisenet.model import BiSeNet
from model.simple_augment import random_apply_affine
from model.stylegan.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train VToonify-D")
        self.parser.add_argument("--iter", type=int, default=200000, help="total training iterations")
        self.parser.add_argument("--epoch", type=int, default=200, help="total training epochs")
        self.parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
        self.parser.add_argument("--batch", type=int, default=4, help="batch sizes for each gpus")
        self.parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--start_iter", type=int, default=0, help="start iteration")
        self.parser.add_argument("--save_every", type=int, default=5000, help="interval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=0, help="when to start saving a checkpoint")
        self.parser.add_argument("--log_every", type=int, default=1000, help="interval of saving a checkpoint")
        
        self.parser.add_argument("--adv_loss", type=float, default=0.01, help="the weight of adv loss")
        self.parser.add_argument("--grec_loss", type=float, default=0.1, help="the weight of mse recontruction loss")
        self.parser.add_argument("--perc_loss", type=float, default=0.01, help="the weight of perceptual loss")
        self.parser.add_argument("--tmp_loss", type=float, default=1.0, help="the weight of temporal consistency loss")
        self.parser.add_argument("--msk_loss", type=float, default=0.0005, help="the weight of attention mask loss")  
        
        self.parser.add_argument("--fix_degree", action="store_true", help="use a fixed style degree")
        self.parser.add_argument("--fix_style", action="store_true", help="use a fixed style image")
        self.parser.add_argument("--fix_color", action="store_false", help="use the original color (no color transfer)")
        self.parser.add_argument("--exstyle_path", type=str, default='checkpoints/Style-A/exstyle_code.npy', help="path of the extrinsic style code")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--pre_train", type=str, default='checkpoints/Style-A/checkpoint.pt', help="path of the style encoder")
        self.parser.add_argument("--pre_train_vtoonify", type=str, default='checkpoints/Style-A/vtoonify.pt', help="path of the style encoder")

        self.parser.add_argument("--encoder_path", type=str, default=None, help="path to the pretrained encoder model")    
        self.parser.add_argument("--direction_path", type=str, default='checkpoints/Style-A/derections.npy', help="path to the editing direction latents")
        self.parser.add_argument("--faceparsing_path", type=str, default='checkpoints/Style-A/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--style_encoder_path", type=str, default='checkpoints/Style-A/encoder.pt', help="path of the style encoder")
        
        self.parser.add_argument("--name", type=str, default='vtoonify_d_cartoon', help="saved model name")
        self.parser.add_argument("--pretrain", action="store_true", help="if true, only pretrain the encoder")

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt                
    

            
# generate paired data and train vtoonify, see Sec. 4.2.2 for the detail
def train(args, generator, discriminator, g_optim, d_optim, g_ema, percept, parsingpredictor, down, pspencoder, directions, styles, device, dataloader, test_dataloader):
    pbar = range(args.epoch)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_epoch, smoothing=0.01, ncols=130, dynamic_ncols=False)

    d_loss = torch.tensor(0.0, device=device)
    g_loss = torch.tensor(0.0, device=device)
    grec_loss = torch.tensor(0.0, device=device)
    gfeat_loss = torch.tensor(0.0, device=device)
    temporal_loss = torch.tensor(0.0, device=device)
    gmask_loss = torch.tensor(0.0, device=device)
    loss_dict = {}
    
    surffix = '_s'
    if args.fix_style:
        surffix += '%03d'%(args.style_id)
    surffix += '_d'
    if args.fix_degree:
        surffix += '%1.1f'%(args.style_degree)
    if not args.fix_color:
        surffix += '_c'

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    

    step = args.start_iter
    for idx in pbar:
        i = idx + args.start_epoch
        
        if i > args.epoch:
            print("Done!")
            break
        
        for bii, bi in enumerate(dataloader):
            step+=1
            i += 1
            source_image = bi['source_image']
            target_image = bi['target_image']
            source_semantics = bi['source_semantics']
            target_semantics = bi['target_semantics']
            source_image_np = bi['source_image_np']
            target_image_np = bi['target_image_np']


            real_input_image = source_image.cuda()
            real_input_semantics = target_semantics.cuda()
            real_output_image = target_image.cuda()
            real_input_image_np = source_image_np.cuda()
            real_output_image_np = target_image_np.cuda()
            

            # sample style degree
            if args.fix_degree:
                d_s = args.style_degree
            else:
                d_s = np.random.randint(0,6) / 5.0
            if args.fix_color:
                weight = [d_s] * 7 + [0] * 11
            else:
                weight = [d_s] * 7 + [1] * 11
            # style degree condition for discriminator
            degree_label = torch.zeros(args.batch, 1).to(device) + d_s
            
            # style index condition for discriminator
            style_ind = torch.randint(0, styles.size(0), (args.batch,)).to(device)
            if args.fix_style:
                style_ind = style_ind * 0 + args.style_id
            # sample pre-saved E_s(s)
            style = styles[style_ind]
            

            xl = pspencoder(real_output_image)  # torch.Size([8, 18, 512])
            xl = g_ema.zplus2wplus(xl) # E_s(x''_down) 
            xl = torch.cat((style[:,0:7], xl[:,7:18]), dim=1).detach() # w'' = concatenate E_s(s) and E_s(x''_down) torch.Size([8, 18, 512])
            
            real_output_image_np_p = F.interpolate(parsingpredictor(2*(F.interpolate(real_output_image_np, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                    scale_factor=0.5, recompute_scale_factor=False).detach()
            inputs = torch.cat((real_output_image_np, real_output_image_np_p/16.), dim=1)
            real_output = g_ema(inputs, xl, d_s).detach()
            # y_tilde = torch.clamp(y_tilde, -1, 1).detach()

            ###### This part is for training discriminator
            
            requires_grad(g_module.encoder, False)
            requires_grad(g_module.fusion_out, False)
            requires_grad(g_module.fusion_skip, False)  
            requires_grad(discriminator, True)


            xl_input = pspencoder(real_input_image)  # torch.Size([8, 18, 512])
            xl_input = g_ema.zplus2wplus(xl_input) # E_s(x''_down) 
            xl_input = torch.cat((style[:,0:7], xl_input[:,7:18]), dim=1).detach() # w'' = concatenate E_s(s) and E_s(x''_down) torch.Size([8, 18, 512])
            
            real_input_image_np_p = F.interpolate(parsingpredictor(2*(F.interpolate(real_input_image_np, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                    scale_factor=0.5, recompute_scale_factor=False).detach()
            inputs_input = torch.cat((real_input_image_np, real_input_image_np_p/16.), dim=1)
            fake_output = generator(inputs_input, xl_input, real_input_image_np, real_input_semantics, d_s)

            fake_pred = discriminator(F.adaptive_avg_pool2d(fake_output, 256), degree_label, style_ind)
            real_pred = discriminator(F.adaptive_avg_pool2d(real_output, 256), degree_label, style_ind)
            
            # L_adv in Eq.(3)
            d_loss = d_logistic_loss(real_pred, fake_pred) * args.adv_loss
            loss_dict["d"] = d_loss
            
            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()    
            
            ###### This part is for training generator (encoder and fusion modules)

            requires_grad(g_module.encoder, True)
            requires_grad(g_module.fusion_out, True)
            requires_grad(g_module.fusion_skip, True)    
            requires_grad(discriminator, False)        

            fake_output, m_Es = generator(inputs_input, xl_input, real_input_image_np, real_input_semantics, d_s, return_mask=True)
            fake_pred = discriminator(F.adaptive_avg_pool2d(fake_output, 256), degree_label, style_ind)
            
            # L_adv in Eq.(3)
            g_loss = g_nonsaturating_loss(fake_pred) * args.adv_loss
            # L_rec in Eq.(2)
            grec_loss = F.mse_loss(fake_output, real_output) * args.grec_loss
            gfeat_loss = percept(F.adaptive_avg_pool2d(fake_output, 512), # 1024 will out of memory
                                F.adaptive_avg_pool2d(real_output, 512)).sum() * args.perc_loss # 256 will get blurry output
    
            # L_msk in Eq.(9)
            gmask_loss = torch.tensor(0.0, device=device)
            if not args.fix_degree or args.msk_loss > 0:
                for jj, m_E in enumerate(m_Es):
                    gd_s = (1 - d_s) ** 2 * 0.9 + 0.1
                    gmask_loss += F.relu(torch.mean(m_E)-gd_s) * args.msk_loss

            loss_dict["g"] = g_loss
            loss_dict["gr"] = grec_loss
            loss_dict["gf"] = gfeat_loss
            loss_dict["msk"] = gmask_loss

            samplein = real_output_image_np.clone().detach()
            sampleout = fake_output.clone().detach()
            sampleout2 = real_output.clone().detach()

            generator.zero_grad()
            (g_loss + grec_loss + gfeat_loss + gmask_loss).backward() 
            g_optim.step()        
               

            loss_reduced = reduce_loss_dict(loss_dict)

            d_loss_val = loss_reduced["d"].mean().item()
            g_loss_val = loss_reduced["g"].mean().item()
            gr_loss_val = loss_reduced["gr"].mean().item()
            gf_loss_val = loss_reduced["gf"].mean().item()
            # tmp_loss_val = loss_reduced["tp"].mean().item()
            msk_loss_val = loss_reduced["msk"].mean().item()

            if get_rank() == 0:
                pbar.set_description(
                    (
                        f"iter: {step:d}; advd: {d_loss_val:.3f}; advg: {g_loss_val:.3f}; mse: {gr_loss_val:.3f}; "
                        f"perc: {gf_loss_val:.3f}; msk: {msk_loss_val:.3f}"
                    )
                )

                if step == 0 or (step+1) % args.log_every == 0 or (step+1) == args.iter:
                    with torch.no_grad():

                        sample = F.interpolate(torch.cat((samplein, F.adaptive_avg_pool2d(sampleout, 256), F.adaptive_avg_pool2d(sampleout2, 256)), dim=0), 256)
                        utils.save_image(
                            sample,
                            f"log/%s/%05d.jpg"%(args.name, (step+1)),
                            nrow=int(args.batch),
                            normalize=True,
                            range=(-1, 1),
                        )

                if ((step+1) >= args.save_begin and (step+1) % args.save_every == 0) or (step+1) == args.iter:
                    if (step+1) == args.iter:
                        savename = f"checkpoints/style-A/%s/vtoonify%s.pt"%(args.name, surffix)
                    else:
                        savename = f"checkpoints/style-A/vtoonify%s_%05d.pt"%(args.name, surffix, step+1) 
                    torch.save(
                        {
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            # "g_ema": g_ema.state_dict(),
                        },
                        savename,
                    )
                    
                
from collections import OrderedDict
if __name__ == "__main__":
    
    device = "cuda"
    parser = TrainOptions()  
    args = parser.parse()
    if args.local_rank == 0:
        print('*'*98)
        if not os.path.exists("log/%s/"%(args.name)):
            os.makedirs("log/%s/"%(args.name))
        if not os.path.exists("checkpoints/Style-A/%s/"%(args.name)):
            os.makedirs("checkpoints/Style-A/%s/"%(args.name))
        
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    
    # args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    generator = Audio_Driven_VToonify(backbone = 'dualstylegan').to(device)
    # generator.apply(weights_init)
    try:
        generator.load_state_dict(torch.load(args.pre_train, map_location=lambda storage, loc: storage)['g_ema'], strict=False)
    except:
        generator.load_state_dict(torch.load(args.pre_train, map_location=lambda storage, loc: storage)['g'])

    
    g_ema = VToonify(backbone = 'dualstylegan').to(device)
    g_ema.load_state_dict(torch.load(args.pre_train_vtoonify, map_location=lambda storage, loc: storage)['g_ema'])
    g_ema.eval()

    requires_grad(generator.generator, False)
    requires_grad(generator.res, False)
    requires_grad(g_ema, False)

    g_parameters = list(generator.encoder.parameters()) 

    g_parameters = g_parameters + list(generator.fusion_out.parameters()) + list(generator.video_warper.parameters()) + list(generator.calibrator.parameters()) + list(generator.fusion_skip.parameters())

    g_optim = optim.Adam(
        g_parameters,
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()
    requires_grad(parsingpredictor, False)

    down = Downsample(kernel=[1, 3, 3, 1], factor=2).to(device)
    requires_grad(down, False)

    directions = torch.tensor(np.load(args.direction_path)).to(device) 


    exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
    if args.local_rank == 0 and not os.path.exists('checkpoints/Style-A/%s/exstyle_code.npy'%(args.name)):
        np.save('checkpoints/style-A/%s/exstyle_code.npy'%(args.name), exstyles, allow_pickle=True)    
    styles = []
    with torch.no_grad(): 
        for stylename in exstyles.keys():
            exstyle = torch.tensor(exstyles[stylename]).to(device)
            exstyle = g_ema.zplus2wplus(exstyle)
            styles += [exstyle]
    styles = torch.cat(styles, dim=0)

    if not args.pretrain:
        discriminator = ConditionalDiscriminator(256, use_condition=True, style_num = styles.size(0)).to(device)
        try:
            discriminator.load_state_dict(torch.load(args.pre_train, map_location=lambda storage, loc: storage)['d'])
        except:
            print('No pre-trained discriminator model!')
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
        )    

        if args.distributed:
            discriminator = nn.parallel.DistributedDataParallel(
                discriminator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[args.local_rank])
        requires_grad(percept.model.net, False)

        pspencoder = load_psp_standalone(args.style_encoder_path, device)  

    if args.local_rank == 0:
        print('Load models and data successfully loaded!')

    opt = Config('config/Style-A/style_a.yaml', is_train=True)

    dataset = StyleADATASET(opt.data, is_inference=False)
    dataloader = DataLoader(dataset, batch_size=int(args.batch), shuffle=True, num_workers=0, drop_last=True)

    test_dataset = StyleADATASET(opt.data, is_inference=True)
    test_dataloader = DataLoader(test_dataset, batch_size=int(args.batch), shuffle=True, num_workers=0, drop_last=True)
    train(args, generator, discriminator, g_optim, d_optim, g_ema, percept, parsingpredictor, down, pspencoder, directions, styles, device, dataloader, test_dataloader)
