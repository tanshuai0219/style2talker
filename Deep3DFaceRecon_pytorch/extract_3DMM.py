import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm, trange
import torch
import face_alignment
from reconstructor import Reconstructor
import copy
from util.preprocess import align_img
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, network_size=4, device='cuda')
face_reconstructor = Reconstructor()
from util.load_mats import load_lm3d


# landmark detection in Deep3DRecon
def lm68_2_lm5(in_lm):
    # in_lm: shape=[68,2]
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    lm = np.stack([in_lm[lm_idx[0],:],np.mean(in_lm[lm_idx[[1,2]],:],0),np.mean(in_lm[lm_idx[[3,4]],:],0),in_lm[lm_idx[5],:],in_lm[lm_idx[6],:]], axis = 0)
    lm = lm[[1,2,0,3,4],:2]
    return lm

lm3d_std = load_lm3d('BFM') 

def image_transform(images, lm):
        W,H = images.size
        if np.mean(lm) == -1:
            lm = (lm3d_std[:, :2]+1)/2.
            lm = np.concatenate(
                [lm[:, :1]*W, lm[:, 1:2]*H], 1
            )
        else:
            lm[:, -1] = H - 1 - lm[:, -1]
        
        trans_params, img, lm, _ = align_img(images, lm, lm3d_std)        
        img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1)
        lm = torch.tensor(lm)
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)])
        # trans_params = torch.tensor(trans_params.astype(np.float32))
        return img, lm, trans_params  

def process_video(fname, out_name=None):
    assert fname.endswith(".mp4")
    if out_name is None:
        out_name = fname[:-4] + '.npy'
    tmp_name = out_name[:-4] + '.doi'
    # if os.path.exists(tmp_name):
    #     print("tmp exist, skip")
    #     return
    # if os.path.exists(out_name):
        # print("out exisit, skip")
        # return
    os.system(f"touch {tmp_name}")
    cap = cv2.VideoCapture(fname)
    lm68_lst = []
    lm5_lst = []
    frame_rgb_lst = []
    trans_param_lst = []
    cnt = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if frame_bgr is None:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            lm68 = fa.get_landmarks(frame_rgb)[0] # 识别图片中的人脸，获得角点, shape=[68,2]
            images = Image.fromarray(frame_rgb)#Image.open(filename).convert('RGB')
            imgs, lms, trans_params = image_transform(images, lm68)
        except:
            print(f"Skip Item: Caught errors when fa.get_landmarks, maybe No face detected in some frames in {fname}!")
            # print(f"Caught error at {cnt}")
            cnt +=1
            return None
            # continue
        lm5 = lm68_2_lm5(lm68)
        trans_param_lst.append(trans_params)
        lm68_lst.append(lm68)
        lm5_lst.append(lm5)
        frame_rgb_lst.append(frame_rgb)
        cnt += 1
    video_rgb = np.stack(frame_rgb_lst) # [t, 224,224, 3]
    lm68_arr = np.stack(lm68_lst).reshape([cnt, 68, 2])
    lm5_arr = np.stack(lm5_lst).reshape([cnt, 5, 2])
    trans_param_arr = np.stack(trans_param_lst).reshape([cnt, 5])
    num_frames = cnt
    batch_size = 32
    iter_times = num_frames // batch_size
    last_bs = num_frames % batch_size
    coeff_lst = []
    for i_iter in range(iter_times):
        start_idx = i_iter * batch_size
        batched_images = video_rgb[start_idx: start_idx + batch_size]
        batched_lm5 = lm5_arr[start_idx: start_idx + batch_size]
        coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
        coeff_lst.append(coeff)
    if last_bs != 0:
        batched_images = video_rgb[-last_bs:]
        batched_lm5 = lm5_arr[-last_bs:]
        coeff, align_img = face_reconstructor.recon_coeff(batched_images, batched_lm5, return_image = True)
        coeff_lst.append(coeff)
    coeff_arr = np.concatenate(coeff_lst,axis=0)
    result_dict = {
        'coeff': coeff_arr.reshape([cnt, -1]),
        'lm68': lm68_arr.reshape([cnt, 68, 2]),
        'lm5': lm5_arr.reshape([cnt, 5, 2]),
        'trans_param': trans_param_arr.reshape([cnt, 5]),
    }
    np.save(out_name, result_dict)
    os.system(f"rm {tmp_name}")


# def split_wav(mp4_name):
#     wav_name = mp4_name[:-4] + '.wav'
#     if os.path.exists(wav_name):
#         return
#     video = VideoFileClip(mp4_name,verbose=False)
#     dur = video.duration
#     audio = video.audio 
#     assert audio is not None
#     audio.write_audiofile(wav_name,fps=16000,verbose=False,logger=None)

if __name__ == '__main__':
    ### Process Single Long video for NeRF dataset
    # video_id = 'May'
    # video_fname = f"data/raw/videos/{video_id}.mp4"
    # out_fname = f"data/processed/videos/{video_id}/coeff.npy"
    # process_video(video_fname, out_fname)

    ### Process short video clips for LRS3 dataset
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--lrs3_path', default='video_dir', help='')
    parser.add_argument('--save_dir', default='video_dir_3DMM', help='')
    parser.add_argument('--process_id', type=int, default=0, help='')
    parser.add_argument('--total_process', type=int, default=1, help='')
    args = parser.parse_args()

    import os, glob
    os.makedirs(args.save_dir, exist_ok=True)
    lrs3_dir = args.lrs3_path
    mp4_name_pattern = os.path.join(lrs3_dir, "*.mp4")
    mp4_names = glob.glob(mp4_name_pattern)
    mp4_names = sorted(mp4_names)
    if args.total_process > 1:
        assert args.process_id <= args.total_process-1
        num_samples_per_process = len(mp4_names) // args.total_process
        if args.process_id == args.total_process-1:
            mp4_names = mp4_names[args.process_id * num_samples_per_process : ]
        else:
            mp4_names = mp4_names[args.process_id * num_samples_per_process : (args.process_id+1) * num_samples_per_process]
    for mp4_name in tqdm(mp4_names, desc='extracting 3DMM...'):
        name = mp4_name.split('/')[-1]
        source = os.path.join(lrs3_dir, name)
        save_path = os.path.join(args.save_dir, name.replace(".mp4", ".npy"))
        # split_wav(mp4_name)
        try:
            process_video(source,out_name=save_path)
        except:
            print('skip')