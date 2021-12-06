# PyTorch adaptation of [Ravens - Transporter Networks](https://github.com/google-research/ravens)

- ### [Original repository (in TensorFlow)](https://github.com/google-research/ravens)
- ### Original Paper: Transporter Networks: Rearranging the Visual World for Robotic Manipulation
  [Project Website](https://transporternets.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/pdf/2010.14406.pdf)&nbsp;&nbsp;•&nbsp;&nbsp;Conference on Robot Learning (CoRL) 2020
  *Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong,<br>Ivan Krasin, Dan Duong, Vikas Sindhwani, Johnny Lee*

### Project Description
Ravens is a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation, with emphasis on pick and place.
It features a Gym-like API with 10 tabletop rearrangement tasks, each with (i) a scripted oracle that provides expert demonstrations (for imitation learning), and (ii) reward functions that provide partial credit (for reinforcement learning).

<img src="https://github.com/google-research/ravens/blob/master/docs/tasks.png" /><br>

(a) **block-insertion**: pick up the L-shaped red block and place it into the L-shaped fixture.<br>
(b) **place-red-in-green**: pick up the red blocks and place them into the green bowls amidst other objects.<br>
(c) **towers-of-hanoi**: sequentially move disks from one tower to another—only smaller disks can be on top of larger ones.<br>
(d) **align-box-corner**: pick up the randomly sized box and align one of its corners to the L-shaped marker on the tabletop.<br>
(e) **stack-block-pyramid**: sequentially stack 6 blocks into a pyramid of 3-2-1 with rainbow colored ordering.<br>
(f) **palletizing-boxes**: pick up homogeneous fixed-sized boxes and stack them in transposed layers on the pallet.<br>
(g) **assembling-kits**: pick up different objects and arrange them on a board marked with corresponding silhouettes.<br>
(h) **packing-boxes**: pick up randomly sized boxes and place them tightly into a container.<br>
(i) **manipulating-rope**: rearrange a deformable rope such that it connects the two endpoints of a 3-sided square.<br>
(j) **sweeping-piles**: push piles of small objects into a target goal zone marked on the tabletop.<br>

Some tasks require generalizing to unseen objects (d,g,h), or multi-step sequencing with closed-loop feedback (c,e,f,h,i,j).


## Installation

**Step 1.** Create a Conda environment with Python 3, then install Python packages:

```shell
make install
```

Or

```shell
cd ~/transporter-nets-torch
conda create --name ravens_torch python=3.7 -y
conda activate ravens_torch
pip install -r requirements.txt
python setup.py install --user
```

**Step 2.** Export environment variables in your terminal

```shell
export RAVENS_ASSETS_DIR=`pwd`/ravens_torch/environments/assets/;
export WORK=`pwd`;
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Getting Started

**Step 1.** Generate training and testing data (saved locally). Note: remove `--disp` for headless mode.

```shell
python ravens_torch/demos.py --disp=True --task=block-insertion --mode=train --n=10
python ravens_torch/demos.py --disp=True --task=block-insertion --mode=test --n=100
```

You can also manually change the parameters in `ravens_torch/demos.py` and then run `make demos` in the shell (see the Makefile if needed).

To run with shared memory, open a separate terminal window and run `python3 -m pybullet_utils.runServer`. Then add `--shared_memory` flag to the command above.

**Step 2.** Train a model e.g., Transporter Networks model. Model checkpoints are saved to the `data/checkpoints` directory. Optional: you may exit training prematurely after 1000 iterations to skip to the next step.

```shell
python ravens_torch/train.py --task=block-insertion --agent=transporter --n_demos=10
```

Likewise for demos, you can run `make train`.

**Step 3.** Evaluate a Transporter Networks agent using the model trained for 1000 iterations. Results are saved locally into `.pkl` files.

```shell
python ravens_torch/test.py --disp=True --task=block-insertion --agent=transporter --n_demos=10 --n_steps=1000
```

Again, `make test` automates it.

**Step 4.** Plot and print results with `make plot` or:

```shell
python ravens_torch/plot.py --disp=True --task=block-insertion --agent=transporter --n_demos=10
```

**Optional.** Track training and validation losses with Tensorboard.

```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```

## Datasets

Download generated train and test datasets from the original authors of the paper:

```shell
wget https://storage.googleapis.com/ravens-assets/block-insertion.zip
wget https://storage.googleapis.com/ravens-assets/place-red-in-green.zip
wget https://storage.googleapis.com/ravens-assets/towers-of-hanoi.zip
wget https://storage.googleapis.com/ravens-assets/align-box-corner.zip
wget https://storage.googleapis.com/ravens-assets/stack-block-pyramid.zip
wget https://storage.googleapis.com/ravens-assets/palletizing-boxes.zip
wget https://storage.googleapis.com/ravens-assets/assembling-kits.zip
wget https://storage.googleapis.com/ravens-assets/packing-boxes.zip
wget https://storage.googleapis.com/ravens-assets/manipulating-rope.zip
wget https://storage.googleapis.com/ravens-assets/sweeping-piles.zip
```

The MDP formulation for each task uses transitions with the following structure:
- **Observations:** raw RGB-D images and camera parameters (pose and intrinsics).
- **Actions:** a primitive function (to be called by the robot) and parameters.
- **Rewards:** total sum of rewards for a successful episode should be =1.
- **Info:** 6D poses, sizes, and colors of objects.

## Pre-Trained Models

1. Download [this archive](https://drive.google.com/file/d/11Lst1KwCL6oT64wto9fI6vZ3oceR-q3c/view?usp=sharing).
2. Extract the weights:
```shell
tar -xvf checkpoints.tar.gz
```
3. Place the weights in the folder `$WORK/data/checkpoints`

You should now be able to run tests as in __Step 3__ above.
