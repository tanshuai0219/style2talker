import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class StyleADATASET(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_facial.json" if is_inference else "train_facial.json"
        list_file = os.path.join(path, "lists",list_file)

        with open(list_file,"r") as f:
            videos = json.load(f)
        with open(os.path.join(path, "lists",'test_facial.json'),"r") as f:
            videos += json.load(f) 
        self.resolution = opt.resolution
        self.HDTF_ldmk_path = 'HDTF/landmark'
        self.semantic_radius = opt.semantic_radius
        self.MEAD_ldmk_path = 'processed_MEAD_front'
        self.video_items, self.person_ids, self.person_id_meads, self.person_ids_emotion = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.idx_by_person_id_emotion = self.group_by_key(self.video_items, key='person_id_emotion')
        self.idx_by_person_id_mead = self.group_by_key(self.video_items, key='person_id_mead')
        self.person_ids = self.person_ids #* 100

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def get_video_index(self, videos):
        video_items = []
        person_ids = []
        person_id_meads = []
        person_ids_emotion = []
        tot_len = len(videos)
        print('loading video_index')
        pbar = tqdm(range(tot_len))
        for i in pbar:
            video  = videos[i]
            video_items.append(self.Video_Item(video))
            splits = video.split('#')
            if len(splits) == 2:
                person_ids.append(splits[0])
                person_id_meads.append('lll')
                person_ids_emotion.append(splits[0]+'#'+splits[1])
            else:
                a,b,c,d = splits
                person_ids.append( a+'#'+b+'#'+c)
                person_id_meads.append(a)
                person_ids_emotion.append(splits[0]+'#'+splits[1])

        person_ids = sorted(person_ids)
        person_id_meads = sorted(person_id_meads)
        person_ids_emotion = sorted(person_ids_emotion)

        return video_items, person_ids, person_id_meads, person_ids_emotion

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        splits = video_name.split('#')
        if len(splits) == 2:
            video_item['video_name'] = video_name
            video_item['person_id'] = video_name.split('#')[0] # M003#angry#030 WDA_DonnaShalala1_000#29
            video_item['person_id_mead'] = 'lll'
            video_item['person_id_emotion'] = video_name.split('#')[0]+'#'+video_name.split('#')[1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], 'length')
                length = int(txn.get(key).decode('utf-8'))
            video_item['num_frame'] = length
            
            return video_item
        else:
            a,b,c,d = splits
            video_item['video_name'] = video_name
            video_item['person_id'] = a+'#'+b+'#'+c # M003#angry#030 WDA_DonnaShalala1_000#29
            video_item['person_id_mead'] = a
            video_item['person_id_emotion'] = video_name.split('#')[0]+'#'+video_name.split('#')[1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], 'length')
                length = int(txn.get(key).decode('utf-8'))
            video_item['num_frame'] = length
            
            return video_item

    def __len__(self):
        return len(self.person_ids)
    def align_face_ldmk(self, filepath, lm):
        """
        :param filepath: str
        :return: PIL Image
        """
        
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        # if type(filepath) == str:
        #     img = PIL.Image.open(filepath)
        # else:
        #     img = PIL.Image.fromarray(filepath)
        img = filepath
        output_size = 256
        transform_size = 256
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        return img
    def __getitem__(self, index):
        data={}
        person_id = self.person_ids[index]
        
        if len(person_id.split('#'))==3:
            person_id_neutral = person_id.split('#')[0]+'#neutral'
            source_video_item = self.video_items[random.choices(self.idx_by_person_id_emotion[person_id_neutral], k=1)[0]]
            
            a,b,c,d = source_video_item['video_name'].split('#')
            source_ldmks = np.load(os.path.join(self.MEAD_ldmk_path, a, 'ldmk',b,c,d +'.npy')) # M026#happy#level_3#010
            target_video_item = self.video_items[random.choices(self.idx_by_person_id_mead[person_id.split('#')[0]], k=1)[0]]
            frame_source, frame_target = self.random_select_frames_source_target(source_video_item,target_video_item)
            a,b,c,d = target_video_item['video_name'].split('#')
            target_ldmks = np.load(os.path.join(self.MEAD_ldmk_path, a, 'ldmk',b,c,d +'.npy')) # M026#happy#level_3#010
            
            if frame_source<len(source_ldmks):
                source_ldmk = source_ldmks[frame_source]
            else:
                source_ldmk = source_ldmks[-1]
            if frame_target<len(target_ldmks):
                target_ldmk = target_ldmks[frame_target]
            else:
                target_ldmk = target_ldmks[-1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(source_video_item['video_name'], frame_source) # M027#neutral#014-0000153
                img_bytes_1 = txn.get(key) 
                key = format_for_lmdb(target_video_item['video_name'], frame_target)
                img_bytes_2 = txn.get(key)
                # print(video_item['video_name'], frame_source, frame_target)
                source_semantics_key = format_for_lmdb(source_video_item['video_name'], 'coeff_3dmm') # M027#neutral#014-0000153
                source_semantics_numpy = np.frombuffer(txn.get(source_semantics_key), dtype=np.float32)

                source_semantics_numpy = source_semantics_numpy.reshape((-1,260)) # video_item['num_frame']

                if source_semantics_numpy.shape[0]<source_video_item['num_frame']:
                    source_semantics_numpy = np.concatenate((source_semantics_numpy,source_semantics_numpy[::-1]), axis=0)
                if source_semantics_numpy.shape[0]> source_video_item['num_frame']:
                    source_semantics_numpy = source_semantics_numpy[:source_video_item['num_frame']]

                # print(video_item['video_name'], frame_source, frame_target)
                target_semantics_key = format_for_lmdb(target_video_item['video_name'], 'coeff_3dmm')
                target_semantics_numpy = np.frombuffer(txn.get(target_semantics_key), dtype=np.float32)

                target_semantics_numpy = target_semantics_numpy.reshape((-1,260)) # video_item['num_frame']

                if target_semantics_numpy.shape[0]<target_video_item['num_frame']:
                    target_semantics_numpy = np.concatenate((target_semantics_numpy,target_semantics_numpy[::-1]), axis=0)
                if target_semantics_numpy.shape[0]> target_video_item['num_frame']:
                    target_semantics_numpy = target_semantics_numpy[:target_video_item['num_frame']]

            img1 = Image.open(BytesIO(img_bytes_1))
            img1_np = np.array(img1)
            img1 = self.align_face_ldmk(img1, source_ldmk)

            img2 = Image.open(BytesIO(img_bytes_2))
            img2_np = np.array(img2)
            img2 = self.align_face_ldmk(img1, target_ldmk)
            data['target_image_np'] = self.transform(img2_np) 
            data['target_image'] = self.transform(img2) 
            data['source_image_np'] = self.transform(img1_np)
            data['source_image'] = self.transform(img1)
            data['target_semantics'] = self.transform_semantic(target_semantics_numpy, frame_target)
            data['source_semantics'] = self.transform_semantic(source_semantics_numpy, frame_source)
        
            return data
        else:
            video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
            
            frame_source, frame_target = self.random_select_frames(video_item)
            ldmks = np.load(os.path.join(self.HDTF_ldmk_path, video_item['video_name']+'.npy'))
            if frame_source<len(ldmks):
                source_ldmk = ldmks[frame_source]
            else:
                source_ldmk = ldmks[-1]
            if frame_target<len(ldmks):
                target_ldmk = ldmks[frame_target]
            else:
                target_ldmk = ldmks[-1]


            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], frame_source)
                img_bytes_1 = txn.get(key) 
                key = format_for_lmdb(video_item['video_name'], frame_target)
                img_bytes_2 = txn.get(key)
                # print(video_item['video_name'], frame_source, frame_target)
                semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
                semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
                # print(semantics_numpy.shape[0])
                # print(int(semantics_numpy.shape[0]/video_item['num_frame']))
                # semantics_numpy = semantics_numpy[:int(semantics_numpy.shape[0]/video_item['num_frame'])*257]
                # print(semantics_numpy.shape[0])
                # if semantics_numpy.shape[0]/260
                semantics_numpy = semantics_numpy.reshape((-1,260)) # video_item['num_frame']
                # semantics_numpy = semantics_numpy[:video_item['num_frame']]
                if semantics_numpy.shape[0]<video_item['num_frame']:
                    semantics_numpy = np.concatenate((semantics_numpy,semantics_numpy[::-1]), axis=0)
                if semantics_numpy.shape[0]> video_item['num_frame']:
                    semantics_numpy = semantics_numpy[:video_item['num_frame']]
                # print(semantics_numpy.shape)

            img1 = Image.open(BytesIO(img_bytes_1))
            img1_np = np.array(img1)
            img1 = self.align_face_ldmk(img1, source_ldmk)

            img2 = Image.open(BytesIO(img_bytes_2))
            img2_np = np.array(img2)
            img2 = self.align_face_ldmk(img1, target_ldmk)
            data['target_image_np'] = self.transform(img2_np) 
            data['target_image'] = self.transform(img2) 
            data['source_image_np'] = self.transform(img1_np)
            data['source_image'] = self.transform(img1)
            data['target_semantics'] = self.transform_semantic(semantics_numpy, frame_target)
            data['source_semantics'] = self.transform_semantic(semantics_numpy, frame_source)
        
            return data
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def random_select_frames_source_target(self, source_video_item, target_video_item):
        num_frame = source_video_item['num_frame']
        from_frame_idx = random.choices(list(range(num_frame)), k=1)
        num_frame_target = target_video_item['num_frame']
        target_frame_idx = random.choices(list(range(num_frame_target)), k=1)
        return from_frame_idx[0], target_frame_idx[0]

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq

