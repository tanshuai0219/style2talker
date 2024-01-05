import os
from unittest import main
from skimage import io, img_as_float32, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob
import pickle
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch, json
import lmdb

def random_select_sentence(name, text_dict):
    sentences = text_dict[name]
    r = random.choice([x for x in range(5)])
    return sentences[r]

class StyleEDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train

        text_file = 'MEAD_text.json'
        with open(text_file,"r") as f:
            self.text_dict = json.load(f)

        if self.is_train:
            self.type = 'train'
            file = 'train.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)
        else:
            self.type = 'test'
            file = 'test.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        name = self.video_list[idx]
        splits = name.split('#')
        a,b,c,d = splits
        npy_path = os.path.join(self.root_dir, a, '3DMM', b, c,d+'.npy')
        coeffs_pred_numpy = np.load(npy_path, allow_pickle=True)
        coeffs_pred_numpy = dict(enumerate(coeffs_pred_numpy.flatten(), 0))[0]
        coeff = coeffs_pred_numpy['coeff']
        coeff_mouth = coeff
        while(len(coeff_mouth)<50):
            coeff_mouth = np.concatenate((coeff_mouth,coeff_mouth[::-1,:]),axis = 0)
        len_ = len(coeff_mouth)
        r = random.choice([x for x in range(3,len_-32)])
        coeff_mouth = coeff_mouth[r:r+32,:]
        parameters = np.array(coeff_mouth)


        style_index = random.choice([x for x in range(len(self.video_list))])
        style_name = self.video_list[style_index]
        style_a = style_name.split('#')[0]
        while (style_a!=a):
            style_index = random.choice([x for x in range(len(self.video_list))])
            style_name = self.video_list[style_index]
            style_a = style_name.split('#')[0]


        style_splits = style_name.split('#')
        style_a,style_b,style_c,style_d = style_splits
        style_npy_path = os.path.join(self.root_dir, style_a, '3DMM', style_b, style_c,style_d+'.npy')
        style_audio_path = os.path.join(self.root_dir, style_a, 'mel', style_b, style_c,style_d+'.npy')
        style_mel = np.load(style_audio_path)

        style_coeffs_pred_numpy = np.load(style_npy_path, allow_pickle=True)
        style_coeffs_pred_numpy = dict(enumerate(style_coeffs_pred_numpy.flatten(), 0))[0]
        style_coeff = style_coeffs_pred_numpy['coeff']
        style_coeff_mouth = style_coeff
        
        style_aus_name = style_a+'_'+style_b+'_'+style_c+'_'+style_d
        style_au_sentence = random_select_sentence(style_aus_name, self.text_dict)

        while(len(style_coeff_mouth)<50):
            style_coeff_mouth = np.concatenate((style_coeff_mouth,style_coeff_mouth[::-1,:]),axis = 0)

        style_len_ = len(style_coeff_mouth)
        style_r = random.choice([x for x in range(3,style_len_-32)])
        style_coeff_mouth = style_coeff_mouth[style_r:style_r+32,:]
        style_mel_feature = style_mel[style_r:style_r+32,:]
        style_parameters = np.array(style_coeff_mouth)


        return parameters, style_au_sentence, style_mel_feature,  style_parameters
