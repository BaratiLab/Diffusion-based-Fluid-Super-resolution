# Diffusion-based-Fluid-Super-resolution
<br>

PyTorch implementation of **A Physics-informed Diffusion Model for High-fidelity Flow Field Reconstruction** 

<div>
<p>arXiv: <a href="https://arxiv.org/abs/2211.14680">link</a></p>
<p>Preprint: Journal of Computational Physics</p>
</div>

<div style style=”line-height: 25%” align="center">
<h3>Sample 1</h3>
<img src="https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/images/reconstruction_sample_01.gif">
<h3>Sample 2</h3>
<img src="https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/images/reconstruction_sample_02.gif">
</div>


## Datasets
Datasets used for model training and sampling can be downloaded via the following links.

- High resolution data (ground truth for super-resolution tasks) (<a href="https://figshare.com/ndownloader/files/39181919">link</a>)

- Low resolution data measured from random grid locations (input data for super-resolution tasks) (<b>link coming soon</b>)


## Running the Experiments
This code has been tested on the following environment
```
PyTorch 1.7

CUDA 10.1

TensorBoard 2.11

Numpy 1.22
```

Download the high res and low res data and save the data files to the ``./data`` directory.

<p>(<font color="blue">  More details about how to run the experiments are coming soon. </font>)</p>

- Step 1 - Model Training

In the directory ``./train_ddpm``, run:

``
bash train.sh
``

or 

``
python main.py --config ./km_re1000_rs256_conditional.yml --exp ./experiments/km256/ --doc ./weights/km256/ --ni
``

- Step 2 - Super-resolution

In the main directory of this repo, run:


``
python main.py --config kmflow_re1000_rs256.yml --seed 1234 --sample_step 1 --t 240 --r 30
``


This implementation is based on / inspired by:

- [https://github.com/ermongroup/SDEdit](https://github.com/ermongroup/SDEdit) (SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations), 
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (Denoising Diffusion Implicit Models)
