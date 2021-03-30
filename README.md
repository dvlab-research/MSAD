# MSAD
**Multi-Scale Aligned Distillation for Low-Resolution Detection**

Lu Qi, Jason Kuen, Jiuxiang Gu, Zhe Lin, Yi Wang, Yukang Chen, Yanwei Li, Jiaya Jia

<!-- [[`arXiv`](https://arxiv.org/pdf/2012.00720.pdf)] [[`BibTeX`](#CitingPanopticFCN)] -->

<div align="center">
  <img src="docs/Framework-crop.png"/>
</div><br/>


This project provides an implementation for the CVPR 2021 paper "[Multi-Scale Aligned Distillation for Low-Resolution Detection]" based on [Detectron2](https://github.com/facebookresearch/detectron2). MSAD target to detect objects in low-resolution image size, with obtaining comparable performance to high-resolution image size. Our paper use [Slimmable Neural Networks](https://arxiv.org/abs/1812.08928) as our pretrained weight.


## Installation
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2), which can be constructed as follows.
* Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).
* Setup the dataset following [the structure](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md).
* Copy this project to `/path/to/detectron2/projects/MSAD`

## Teacher Training
To train teacher model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/MSAD/train_T.py --config-file <projects/MSAD/configs/config.yaml> --num-gpus 8
```

For example, to launch MSAD teacher training (1x schedule) with Slimmable-ResNet-50 backbone in 0.25 width on 8 GPUs,
one should execute:
```bash
cd /path/to/detectron2
python3 projects/MSAD/train_T.py --config-file projects/MSAD/configs/MSAD-R50-T025-1x.yaml --num-gpus 8 OUTPUT_DIR MSAD-R50-T025-1x
```

## Student Training
To train student model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/MSAD/train_S.py --config-file <projects/MSAD/configs/config.yaml> --num-gpus 8
```

For example, to launch MSAD student training (1x schedule) with Slimmable-ResNet-50 backbone in 0.25 width on 8 GPUs,
one should execute:
```bash
cd /path/to/detectron2
python3 projects/MSAD/train_S.py --config-file projects/MSAD/configs/MSAD-R50-S025-1x.yaml --num-gpus 8 MODEL.WEIGHTS MSAD-R50-T025-1x/latest_model.pth OUTPUT_DIR MSAD-R50-S025-1x
```

## Evaluation
To evaluate a pre-trained model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/MSAD/train.py --config-file <config.yaml> --num-gpus 8 --eval-only MODEL.WEIGHTS MSAD-R50-S025-1x/model_checkpoint
```