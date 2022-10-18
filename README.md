# Self-Supervised Vision Transformers with DINO

PyTorch implementation and pretrained models for DINO. For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`blogpost`](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)] [[`arXiv`](https://arxiv.org/abs/2104.14294)] [[`Yannic Kilcher's video`](https://www.youtube.com/watch?v=h3ij3F3cPIk)]

<div align="center">
  <img width="100%" alt="DINO illustration" src=".github/dino.gif">
</div>

## Setup
- Install [conda](https://docs.conda.io/en/latest/miniconda.html).
- Create conda environment by running `conda env create`.

## Evaluate model on semantic segmentation propagation

### DINO

This assumes you have a folder of a run with rgb images and semantic segmentation labels `${DATAROOT}`, and
the weights of a pre-trained DINO model `${PRETRAINED_WEIGHTS}`.

To predict the segmentation labels

- Activate the conda environment with `conda activate dino`
- Run `python eval_ipace_segmentation.py --pretrained_weights ${PRETRAINED_WEIGHTS} --data_path ${DATAROOT} --output_dir 
${OUTPUT_DIR}` to generate the predictions.

This will create folder `${OUTPUT_DIR}` with the predicted segmentations.

To evaluate the intersection-over-union metrics of the predictions, and visualise them:

- Run `python compute_segmentation_metrics.py --dataset_path ${DATAROOT} --prediction_path ${OUTPUT_DIR}`


### Masked autoencoder
For the predictions of masked autoencoder, clone the repository `git clone git@github.com:anthonyhu/mae.git`

To predict the segmentations:

- `python eval_ipace_segmentation.py --pretrained_weights ${PRETRAINED_WEIGHTS} --data_path ${DATAROOT} --output_dir 
${OUTPUT_DIR} --path_to_mae_repo ${MAE_REPO_PATH} --arch mae`

To evaluate the intersection-over-union metrics of the predictions, and visualise them:

- Run `python compute_segmentation_metrics.py --dataset_path ${DATAROOT} --prediction_path ${OUTPUT_DIR}`



## Training

### Documentation
Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). For a glimpse at the full documentation of DINO training please run:
```
python main_dino.py --help
```

### Vanilla DINO training :sauropod:
Run DINO with ViT-small network on a single node with 8 GPUs for 100 epochs with the following command. Training time is 1.75 day and the resulting checkpoint should reach 69.3% on k-NN eval and 74.0% on linear eval. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_vanilla_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.
```
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

### Multi-node training
We use Slurm and [submitit](https://github.com/facebookincubator/submitit) (`pip install submitit`). To train on 2 nodes with 8 GPUs each (total 16 GPUs):
```
python run_with_submitit.py --nodes 2 --ngpus 8 --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

<details>
<summary>
DINO with ViT-base network.
</summary>

```
python run_with_submitit.py --nodes 2 --ngpus 8 --use_volta32 --arch vit_base  --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

### Boosting DINO performance :t-rex:
You can improve the performance of the vanilla run by:
- training for more epochs: `--epochs 300`,
- increasing the teacher temperature: `--teacher_temp 0.07 --warmup_teacher_temp_epochs 30`.
- removing last layer normalization (only safe with `--arch vit_small`): `--norm_last_layer false`,

<details>
<summary>
Full command.
</summary>

```
python run_with_submitit.py --arch vit_small --epochs 300 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --norm_last_layer false --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

</details>

The resulting pretrained model should reach 73.3% on k-NN eval and 76.0% on linear eval. Training time is 2.6 days with 16 GPUs. We provide [training](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_log.txt) and [linear evaluation](https://dl.fbaipublicfiles.com/dino/example_runs_logs/dino_boost_deitsmall16_eval.txt) logs (with batch size 256 at evaluation time) for this run to help reproducibility.
