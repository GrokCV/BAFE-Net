# Background Semantics Matter: Cross-Task Feature Exchange Network for Clustered Infrared Small Target Detection With Sky-Annotated Dataset


- [Installation](#installation)
  - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
  - [Step 2: Install PyTorch](#step-2-install-pytorch)
  - [Step 3: Install OpenMMLab Codebases](#step-3-install-openmmlab-codebases)
  - [Step 4: Install `deepir`](#step-4-install-deepir)
- [Dataset Preparation](#dataset-preparation)
  - [File Structure](#file-structure)
  - [Datasets Link](#datasets-link)
- [Train](#train)
- [Test](#test)
- [Model Zoo and Benchmark](#model-zoo-and-benchmark)
  - [Leaderboard](#leaderboard)
  - [Model Zoo](#model-zoo)


## Installation

Step 1: Create a conda environment

```shell
$ conda create --name deepir python=3.9
$ conda activate deepir
```

Step 2: Install PyTorch

```shell
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Step 3: Install OpenMMLab Codebases

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
pip install "mmsegmentation>=1.0.0"
pip install dadaptation
```

Step 4: Install `deepir`

```shell
$ python setup.py develop
```

**Note**: make sure you have `cd` to the root directory of `deepinfrared`

```shell
$ git clone git@github.com:GrokCV/BAFE-Net.git
$ cd BAFE-Net
```


## Dataset Preparation

### File Structure
```angular2html
|- data
    |- SIRSTdevkit
        |-PNGImages
            |-Misc_1.png
            ......
        |-SIRST
            |-BBox
                |-Misc_1.xml
                ......
            |-BinaryMask
                |-Misc_1_pixels0.png
                |-Misc_1.xml
                ......
            |-PaletteMask
                |-Misc_1.png
                ......
            |-Point_label
                |-Misc_1_pixels0.txt
                ......
        |-SkySeg
            |-BinaryMask
                |-Misc_1_pixels0.png
                |-Misc_1.xml
                ......
            |-PaletteMask
                |-Misc_1.png
                ......
        |-Splits
            |-train_v2.txt
            |-test_v2.txt
            ......
```

- PNGImages is the folder for storing all images.
- SIRST and SkySeg are folders for storing annotation files.
    - SIRST corresponds to infrared small targets.
    - SkySeg corresponds to sky segmentation.

Please make sure that the path of your data set is consistent with the `data_root` in `configs/detection/_base_/datasets/sirst_det_seg_voc_skycp.py`

### Datasets Link

https://drive.google.com/uc?export=download&id=1PY0d1WuCjf_3wAIjDSNhYxREVK27OLzl


## Train

```shell
$ CUDA_VISIBLE_DEVICES=0 python train.py <CONFIG_FILE>
```

For example:

```shell
$ CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/fcos_changer_seg/fcos_changer_seg_r50-caffe_fpn_gn-head_1x_densesirst.py
```

## Test

```shell
$ CUDA_VISIBLE_DEVICES=0 python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>
```

For example:

```shell
$ CUDA_VISIBLE_DEVICES=0 python tools/test_det.py configs/detection/fcos_changer_seg/fcos_changer_seg_r50-caffe_fpn_gn-head_1x_densesirst.py work_dirs/fcos_changer_seg_r50-caffe_fpn_gn-head_1x_densesirst/20240719_162542/best_pascal_voc_mAP_epoch_8.pth
```

If you want to visualize the result, you only add ```--show``` at the end of the above command.

The default image save path is under <SEG_CHECKPOINT_FILE>. You can use `--work-dir` to specify the test log path, and the image save path is under this path by default. Of course, you can also use `--show-dir` to specify the image save path.


## Model Zoo and Benchmark

### Leaderboard

| Method | Backbone | mAP<sub>07</sub>↑ | recall<sub>07</sub>↑ | mAP<sub>12</sub>↑ | recall<sub>12</sub>↑ | Flops↓ | Params↓ |
| - | - | - | - | - | - | - | - |
| **One-stage** | | | | | | | |            |
| FCOS                | ResNet50 | 0.232 | 0.315 | 0.204 | 0.324 | 50.291G | 32.113M |
| SSD                 |          | 0.211 | 0.421 | 0.178 | 0.424 | 87.552G | 23.746M |
| GFL                 | ResNet50 | 0.253 | 0.332 | 0.230 | 0.317 | 52.296G | 32.258M |
| ATSS                | ResNet50 | 0.248 | 0.327 | 0.202 | 0.326 | 51.504G | 32.113M |
| CenterNet           | ResNet50 | 0.000 | 0.000 | 0.000 | 0.000 | 50.278G | 32.111M |
| PAA                 | ResNet50 | 0.255 | 0.545 | 0.228 | 0.551 | 51.504G | 32.113M |
| PVT-T               |          | 0.109 | 0.481 | 0.093 | 0.501 | 41.623G | 21.325M |
| RetinaNet           | ResNet50 | 0.114 | 0.510 | 0.086 | 0.523 | 52.203G | 36.330M |
| EfficientDet        |          | 0.099 | 0.433 | 0.072 | 0.419 | 34.686G | 18.320M |
| TOOD                | ResNet50 | 0.256 | 0.355 | 0.226 | 0.342 | 50.456G | 32.018M |
| VFNet               | ResNet50 | 0.253 | 0.336 | 0.214 | 0.336 | 48.317G | 32.709M |
| YOLOF               | ResNet50 | 0.091 | 0.009 | 0.002 | 0.009 | 25.076G | 42.339M |
| AutoAssign          | ResNet50 | 0.255 | 0.354 | 0.180 | 0.314 | 50.555G | 36.244M |
| DyHead              | ResNet50 | 0.249 | 0.335 | 0.189 | 0.328 | 27.866G | 38.890M |
| **Two-stage** | | | | | | | |
| Faster R-CNN        | ResNet50 | 0.091 | 0.022 | 0.015 | 0.029 | 0.759T  | 33.035M |
| Cascade R-CNN       | ResNet50 | 0.136 | 0.188 | 0.139 | 0.194 | 90.978G | 69.152M |
| Dynamic R-CNN       | ResNet50 | 0.184 | 0.235 | 0.111 | 0.190 | 63.179G | 41.348M |
| Grid R-CNN          | ResNet50 | 0.091 | 0.018 | 0.025 | 0.037 | 0.177T  | 64.467M |
| Libra R-CNN         | ResNet50 | 0.141 | 0.142 | 0.085 | 0.120 | 63.990G | 41.611M |
| **End2End** | | | | | | | |
| DETR                | ResNet50 | 0.000 | 0.000 | 0.000 | 0.000 | 24.940G | 41.555M |
| Deformable DETR     | ResNet50 | 0.024 | 0.016 | 0.018 | 0.197 | 51.772G | 40.099M |
| DAB-DETR            | ResNet50 | 0.005 | 0.054 | 0.000 | 0.001 | 28.939G | 43.702M |
| Conditional DETR    | ResNet50 | 0.000 | 0.000 | 0.000 | 0.001 | 27.143G | 40.297M |
| Sparse R-CNN        | ResNet50 | 0.183 | 0.572 | 0.154 | 0.614 | 45.274G | 0.106G  |
| **BAFE-Net (Ours)** | ResNet50 | **0.270** | 0.332 | **0.236** | 0.329 | 69.114G | 35.329M |

### Model Zoo
Checkpoint and Train log: https://drive.google.com/uc?export=download&id=1_heou4B5w5htte8R7eUhyptGe2ZIa-g7

