# $\text{Style}^2\text{Talker}$: High-Resolution Talking Head Generation with Emotion Style and Art Style

#### This repository provides official implementations of PyTorch for the $partial$ core components of the following paper:<br>
[**$\text{Style}^2\text{Talker}$: High-Resolution Talking Head Generation with Emotion Style and Art Style**](https://ojs.aaai.org/index.php/AAAI/article/view/28313)<br>
[Shuai Tan](https://scholar.google.com.hk/citations?user=9KjKwDwAAAAJ&hl=zh-CN), et al.<br>
In AAAI, 2024.<br>


![visualization](demo/teaser.svg)

Our approach takes an identity image and an audio clip as inputs and generates a talking head with emotion style and art style, which are controlled respectively by an emotion source text and an art source picture. The pipeline of our $\text{Style}^2\text{Talker}$ is as follows:

![visualization](demo/pipeline.svg)


## Requirements
We train and test based on Python 3.7 and Pytorch. To install the dependencies run:
```
conda create -n style2talker python=3.7
conda activate style2talker
```

- python packages
```
pip install -r requirements.txt
```

## Inference

- Run the demo：
    ```bash
    python inference.py --img_path path/to/image --wav_path path/to/audio --source_3DMM path/to/source_3DMM --style_e_source "a textual description for emotion style" --art_style_id num/for/art_style --save_path path/to/save
    ```
  The result will be stored in save_path.


## Data Preprocess:
- Crop videos in training datasets:
    ```bash
    python data_preprocess/crop_video.py
    ```
- Extract 3DMM parameters from cropped videos using [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction):
    ```bash
    python data_preprocess/extract_3DMM.py
    ```
- Extract landmarks from cropped videos:
    ```bash
    python data_preprocess/extract_lmdk.py
    ```
- Extract mel feature from audio:
    ```bash
    python data_preprocess/get_mel.py
    ```
- We save the video frames and 3DMM parameters in a lmdb file:
    ```bash
    python data_preprocess/prepare_lmdb.py
    ```
## Train
- Following VToonify, different art styles correspond to different checkpoints, and you can use the following script to train the model to get the art style you want:
    ```bash
    # Train Style-A:
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 12344 train_style_a.py
    ```
    
## Dataset
- We use the following dataset for Style-E training.
1) **MEAD**. [download link](https://wywu.github.io/projects/MEAD/MEAD.html).
- We use the following dataset for Style-A training.
1) **MEAD**. [download link](https://wywu.github.io/projects/MEAD/MEAD.html).
2) **HDTF**. [download link](https://github.com/MRzzm/HDTF).
- Art reference picture dataset.
1) **Cartoon**. [download link](https://mega.nz/file/HslSXS4a#7UBanJTjJqUl_2Z-JmAsreQYiJUKC-8UlZDR0rUsarw).
2) **Illustration, Arcane, Comic, Pixar**. [download link](https://github.com/williamyang1991/DualStyleGAN/tree/main).


## Acknowledgement
Some code are borrowed from following projects:
* [AGRoL](https://github.com/facebookresearch/AGRoL)
* [PIRenderer](https://github.com/RenYurui/PIRender)
* [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
* [SadTalker](https://github.com/OpenTalker/SadTalker)
* [VToonify](https://github.com/williamyang1991/VToonify)
* [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN)
* [StyleHEAT](https://github.com/OpenTalker/StyleHEAT)
* [FOMM video preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing)

Thanks for their contributions!

## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@inproceedings{tan2024style2talker,
  title={Style2Talker: High-Resolution Talking Head Generation with Emotion Style and Art Style},
  author={Tan, Shuai and Ji, Bin and Pan, Ye},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={5079--5087},
  year={2024}
}
```