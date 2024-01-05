# $\text{Style}^2\text{Talker}$: High-Resolution Talking Head Generation with Emotion Style and Art Style

This repository provides the official PyTorch implementation for the following paper:<br>
**$\text{Style}^2\text{Talker}$: High-Resolution Talking Head Generation with Emotion Style and Art Style**<br>
[Shuai Tan](https://scholar.google.com.hk/citations?user=9KjKwDwAAAAJ&hl=zh-CN), et al.<br>
In AAAI, 2024.<br>


![visualization](demo/teaser.svg)

Our approach takes an identity image and an audio clip as inputs and generates a talking head with emotion style and art style, which are controlled respectively by an emotion source text and an art source picture. The pipeline of our $\text{Style}^2\text{Talker}$ is as follows:

![visualization](demo/pipeline.svg)

To text-driven emotion style generation in the case of the text-emotion paired data scarcity, we present a labor-free approach that relies on large-scale pretrained models to automatically generate corresponding emotional textual descriptions for videos in an existing emotional audio-visual dataset:

![visualization](demo/data.svg)

## Large-scale pretrained models for labor-free automatically annotation pipeline
* [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
* [GPT-3](https://github.com/openai/GPT-3)
* [CLIP](https://github.com/openai/CLIP)



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


## Evaluation
- We use [VToonify](https://github.com/williamyang1991/VToonify) to generate artistically stylized ground truth from MEAD and HDTF, and adopt the codes released by [vico_challenge_baseline](https://github.com/dc3ea9f/vico_challenge_baseline/tree/main/evaluations) to assess the results.

## Data Preprocess:
- Crop videos in training datasets:
    ```bash
    python data_preprocess/crop_video.py
    ```
- Split video: Since the video in HDTF is too long, we split both the video and the corresponding audio into 5s segments:
    ```bash
    python data_preprocess/split_HDTF_video.py
    ```

    ```bash
    python data_preprocess/split_HDTF_audio.py
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
- Train Style-E:
    ```bash
    python train_style_e.py
    ```
- Train Style-A:
    ```bash
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
