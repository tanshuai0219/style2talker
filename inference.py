import os
import argparse
import numpy as np
import cv2
import dlib
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import Audio_Driven_VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import load_psp_standalone
import utils.video_util as video_util
import clip
from utils.parser_util import sample_args
import random
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from models.audio_encoder import *


from PIL import Image

def obtain_seq_index(index, num_frames):
    seq = list(range(index - 13, index + 13 + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

def transform_semantic(semantic, frame_index):
    # 500, 73
    index = obtain_seq_index(frame_index, semantic.shape[0])
    coeff_3dmm = semantic[index, ...]
    # 27, 73
    return torch.Tensor(coeff_3dmm).permute(1, 0)

def load_diffusion_model(args):
    print("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to("cuda:0")  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion

def non_overlapping_test(
    args,
    source,
    sample_fn,
    style_code,
    model,
    num_per_batch=256,
    model_type="mlp",
):

    num_frames = source.shape[1]

    output_samples = []
    count = 0
    sparse_splits = []
    flag_index = None

    if args.input_motion_length <= num_frames:
        while count < num_frames:
            if count + args.input_motion_length > num_frames:
                tmp_k = num_frames - args.input_motion_length
                sub_sparse = source[
                    :, tmp_k : tmp_k + args.input_motion_length
                ]
                flag_index = count - tmp_k
            else:
                sub_sparse = source[
                    :, count : count + args.input_motion_length
                ]
            sparse_splits.append(sub_sparse)
            count += args.input_motion_length
    else:
        flag_index = args.input_motion_length - num_frames
        tmp_init = source[:, :1].repeat(1, flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, source], dim=1)
        sparse_splits = [sub_sparse]

    n_steps = len(sparse_splits) // num_per_batch
    if len(sparse_splits) % num_per_batch > 0:
        n_steps += 1
    # Split the sequence into n_steps non-overlapping batches

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    for step_index in range(n_steps):
        sparse_per_batch = torch.cat(
            sparse_splits[
                step_index * num_per_batch : (step_index + 1) * num_per_batch
            ],
            dim=0,
        )

        new_batch_size = sparse_per_batch.shape[0]

        if model_type == "DiffMLP":
            sample = sample_fn(
                model,
                (new_batch_size, args.input_motion_length, args.motion_nfeat),
                sparse=sparse_per_batch,
                style_code = style_code.unsqueeze(1).repeat(1,sparse_per_batch.shape[1],1),
                clip_denoised=False,
                model_kwargs=None,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=noise,
                const_noise=False,
            )
        elif model_type == "mlp":
            sample = model(sparse_per_batch)

        if flag_index is not None and step_index == n_steps - 1:
            last_batch = sample[-1]
            last_batch = last_batch[flag_index:]
            sample = sample[:-1].reshape(-1, args.motion_nfeat)
            sample = torch.cat([sample, last_batch], dim=0)
        else:
            sample = sample.reshape(-1, args.motion_nfeat)
    


    return sample.cpu().float()


def overlapping_test(
    args,
    data,
    sample_fn,
    dataset,
    model,
    sld_wind_size=70,
    model_type="diffusion",
):
    assert (
        model_type == "diffusion"
    ), "currently only diffusion model supports overlapping test!!!"

    gt_data, sparse_original, body_param, head_motion, filename = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.cuda().float()
    sparse_original = sparse_original.cuda().float()
    head_motion = head_motion.cuda().float()
    num_frames = head_motion.shape[0]

    output_samples = []
    count = 0
    sparse_splits = []
    flag_index = None

    if num_frames < args.input_motion_length:
        flag_index = args.input_motion_length - num_frames
        tmp_init = sparse_original[:, :1].repeat(1, flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, sparse_original], dim=1)
        sparse_splits = [sub_sparse]

    else:
        while count + args.input_motion_length <= num_frames:
            if count == 0:
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
                tmp_idx = 0
            else:
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
                tmp_idx = args.input_motion_length - sld_wind_size
                
            sparse_splits.append([sub_sparse, tmp_idx])
            count += sld_wind_size

        if count < num_frames:
            sub_sparse = sparse_original[:, -args.input_motion_length :]
            tmp_idx = args.input_motion_length - (
                num_frames - (count - sld_wind_size + args.input_motion_length)
            )
            sparse_splits.append([sub_sparse, tmp_idx])

    memory = None  # init memory

    if args.fix_noise:
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1).cuda()
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    for step_index in range(len(sparse_splits)):
        sparse_per_batch = sparse_splits[step_index][0]
        memory_end_index = sparse_splits[step_index][1]

        new_batch_size = sparse_per_batch.shape[0]
        assert new_batch_size == 1

        if memory is not None:
            model_kwargs = {}
            model_kwargs["y"] = {}
            model_kwargs["y"]["inpainting_mask"] = torch.zeros(
                (
                    new_batch_size,
                    args.input_motion_length,
                    args.motion_nfeat,
                )
            ).cuda()
            model_kwargs["y"]["inpainting_mask"][:, :memory_end_index, :] = 1
            model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                (
                    new_batch_size,
                    args.input_motion_length,
                    args.motion_nfeat,
                )
            ).cuda()
            model_kwargs["y"]["inpainted_motion"][:, :memory_end_index, :] = memory[
                :, -memory_end_index:, :
            ]
        else:
            model_kwargs = None

        sample = sample_fn(
            model,
            (new_batch_size, args.input_motion_length, args.motion_nfeat),
            sparse=sparse_per_batch,
            clip_denoised=False,
            model_kwargs=None,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=noise,
            const_noise=False,
        )

        memory = sample.clone().detach()

        if flag_index is not None:
            sample = sample[:, flag_index:].cpu().reshape(-1, args.motion_nfeat)
        else:
            sample = sample[:, memory_end_index:].reshape(-1, args.motion_nfeat)

        if not args.no_normalization:
            output_samples.append(dataset.inv_transform(sample.cpu().float()))
        else:
            output_samples.append(sample.cpu().float())

    return output_samples, body_param, head_motion, filename

def add_audio(video_name=None, audio_dir = None):

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
    print (command)
    os.system(command)
    os.remove(video_name)

from collections import OrderedDict
import json

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default=None)
        self.parser.add_argument("--art_style_id", type=int, default=0, help="the id of the art style image")
        self.parser.add_argument("--art_style_degree", type=float, default=0.5, help="art style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_fasle", help="transfer the color of the art style")
        self.parser.add_argument("--art_ckpt", type=str, default='checkpoints/Style-A/Audio_driven_vtoonofy.pt', help="path of the saved model")
        self.parser.add_argument("--scale_image", action="store_false", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--art_style_encoder_path", type=str, default='checkpoints/Style-A/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--art_exstyle_path", type=str, default='checkpoints/Style-A/exstyle_code.npy', help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='checkpoints/Style-A/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        self.parser.add_argument("--batch_size", type=int, default=4, help="batch size of frames when processing video")
        self.parser.add_argument("--parsing_map_path", type=str, default=None, help="path of the refined parsing map of the target video")

        self.parser.add_argument("--emotion_ckpt", type=str, default='checkpoints/Style-E/checkpoint.pth')
        self.parser.add_argument("--batch_size", type=int, default=128)
        self.parser.add_argument("--motion_nfeat", type=int, default=64)
        self.parser.add_argument("--sparse_dim", type=int, default=64)
        self.parser.add_argument("--input_motion_length", type=int, default=32)
        self.parser.add_argument("--latent_dim", type=int, default=512)
        self.parser.add_argument("--emotion_style_code_dim", type=int, default=512)
        self.parser.add_argument("--style_e_config", type=str, default='config/Style-E/style_e.json')
        self.parser.add_argument("--audioencoder_ckpt", type=str, default=None)
        
        self.parser.add_argument("--wav_path", type=str, default='demo/source/audio/test.mp3')
        self.parser.add_argument("--pose_path", type=str, default='demo/source/pose/test.npy')
        self.parser.add_argument("--source_3DMM", type=str, default='demo/source/image_3DMM/test.npy')
        self.parser.add_argument("--save_path", type=str, default='demo/results/test.mp4')
        self.parser.add_argument("--image_path", type=str, default='demo/source/image/test.jpg')
        self.parser.add_argument("--style_e_source", type=str, default='The person is speaking joyfully, pulling up the lip corners.')



    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args_diff = sample_args(args.emotion_ckpt)
    args_diff.model_path = args.emotion_ckpt 

    args_diff.timestep_respacing = "ddim5"

    torch.backends.cudnn.benchmark = False
    random.seed(args_diff.seed)
    np.random.seed(args_diff.seed)
    torch.manual_seed(args_diff.seed)


    model_type = args_diff.arch.split("_")[0]

    args_diff.batch_size  =  args.batch_size
    args_diff.motion_nfeat  = args.motion_nfeat
    args_diff.sparse_dim  = args.sparse_dim
    args_diff.input_motion_length  = args.input_motion_length
    args_diff.latent_dim  =  args.latent_dim
    args_diff.style_code_dim = args.emotion_style_code_dim
    args_diff.config  =  args.style_e_config

    model, diffusion = load_diffusion_model(args_diff)
    sample_fn = diffusion.p_sample_loop
    

    if not args_diff.overlapping_test:
        test_func = non_overlapping_test
        n_testframe = args_diff.num_per_batch
    else:
        print("Overlapping testing...")
        test_func = overlapping_test
        n_testframe = args_diff.sld_wind_size

    
    audioencoder = AudioEncoder()
    audioencoder_ckpt = torch.load(args.audioencoder_ckpt)
    audioencoder.load_state_dict(audioencoder_ckpt)
    audioencoder = audioencoder.to(device)
    audioencoder.eval()

    
    text_encoder, preprocess = clip.load("ViT-B/32", device=device)
    text_encoder.eval()


    audio_driven_vtoonify = Audio_Driven_VToonify(backbone = args.backbone)
    audio_driven_vtoonify.load_state_dict(torch.load(args.art_ckpt, map_location=lambda storage, loc: storage)['g'])
    audio_driven_vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        transforms.Resize([256,256])
        ])

    modelname = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    landmarkpredictor = dlib.shape_predictor(modelname)

    pspencoder = load_psp_standalone(args.art_style_encoder_path, device)    

    print('model loaded!')


    wav_path = args.wav_path 
    pose_path = args.pose_path
    source_3DMM = args.source_3DMM
    save_path = args.save_path
    image_path = args.image_path
    sentences = [args.style_e_source]


    args.content = image_path
    text = clip.tokenize(sentences).to(device)
    text_features = text_encoder.encode_text(text)

    source_coeffs_pred_numpy = np.load(source_3DMM, allow_pickle=True)
    source_coeffs_pred_numpy = dict(enumerate(source_coeffs_pred_numpy.flatten(), 0))[0]
    source_coeff = source_coeffs_pred_numpy['coeff']
    source_coeff_mouth = source_coeff

    source_example_parameters = torch.from_numpy(np.array(source_coeff_mouth[0])).unsqueeze(0).cuda()
    source_audio_feature, source_nums = get_mel(wav_path)

    source_example_parameters = source_example_parameters.type(torch.FloatTensor).cuda()
    source_audio_feature = source_audio_feature.type(torch.FloatTensor).cuda()


    source_predict = audioencoder(source_audio_feature, source_example_parameters[:,80:144].unsqueeze(1))

    output = test_func(
        args_diff,
        source_predict,
        sample_fn,
        text_features,
        model,
        n_testframe,
        model_type=model_type,
    )


    
    image_pil = Image.open(image_path)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    input_image = image_transform(image_pil)
    input_image = input_image.unsqueeze(0).cuda()

    exstyles = np.load(args.art_exstyle_path, allow_pickle='TRUE').item()
    stylename = list(exstyles.keys())[args.art_style_id]
    exstyle = torch.tensor(exstyles[stylename]).to(device)
    with torch.no_grad():  
        exstyle = audio_driven_vtoonify.zplus2wplus(exstyle)

    if args.parsing_map_path is not None:
        x_p_hat = torch.tensor(np.load(args.parsing_map_path))          
            

    filename = args.content
    basename = os.path.basename(filename).split('.')[0]
    scale = 1
    kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])


    frame = cv2.imread(filename)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        I = align_face(frame, landmarkpredictor)
        I = transform(I).unsqueeze(dim=0).to(device)
        s_w = pspencoder(I)
        s_w = audio_driven_vtoonify.zplus2wplus(s_w)
        if args.color_transfer:
            s_w = exstyle
        else:
            s_w[:,:7] = exstyle[:,:7]

        x = transform(frame).unsqueeze(dim=0).to(device)

        x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                            scale_factor=0.5, recompute_scale_factor=False).detach()

        inputs = torch.cat((x, x_p/16.), dim=1)


        coeffs_pred_numpy = np.load(pose_path, allow_pickle=True) # 253*73
        coeffs_pred_numpy = dict(enumerate(coeffs_pred_numpy.flatten(), 0))[0]
        coeff = coeffs_pred_numpy['coeff']
        transform_params = coeffs_pred_numpy['trans_param']
        _, _, ratio, t0, t1 = np.hsplit(transform_params.astype(np.float32), 5)
        coeff_3dmm_cat = np.concatenate([coeff, ratio, t0, t1], 1)

        output = output.cpu().numpy()
        frame_num = output.shape[0]
        coeff_3dmm_cat = coeff_3dmm_cat[:frame_num]


        coeff_3dmm = np.concatenate([output,coeff_3dmm_cat[:,224:227],coeff_3dmm_cat[:,254:257],coeff_3dmm_cat[:,257:300]],1)
        frames = len(coeff_3dmm)
        driven_3dmm = []
        for frame_index in range(frames):
            driven_3dmm.append(transform_semantic(coeff_3dmm, frame_index).cuda())

        video = []
        for frame_index in range(frames):
            y_tilde = audio_driven_vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1),input_image, driven_3dmm[frame_index].unsqueeze(dim=0), d_s = args.art_style_degree)        
            y_tilde = torch.clamp(y_tilde, -1, 1)
            video.append(y_tilde)
        video = torch.cat(video, 0)
        video_util.write2video(save_path, video)
        print('Save video in {}.mp4'.format(save_path))
        add_audio(save_path, wav_path)

