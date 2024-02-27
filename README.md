# DiffTrip

This folder provides a reference implementation of DiffTrip, as described in the paper: "Replace and Refine: Faithful Trip Recommendation with Explicit Diffusion Guidance", which, submitted to KDD 2024 for anonymous review.

## Brief Introduction

DiffTrip leverages Denoising Diffusion Probabilistic Models (DDPMs) to gradually align the generated trajectory with the tourist’s intent. The core idea stems from one of the characteristics of DDPM: the progressive data generation process. We propose & employ an explicit condition-injecting strategy during the inference stage to achieve the alignment. This strategy progressively substitutes the source & destination of the generated trajectory with the ground truth of the source & destination (from the tourist’s query), enabling the model to iteratively refine itself and ultimately produce realistic, intent-consistent trajectories. 

## Environmental Requirements

We run the code on a computer with RTX3060, i5 12400F, and 16G memory. Please Install the dependencies via anaconda:

### Create virtual environment

```
conda create -n DiffTrip python=3.9.18
```

### Activate environment

```
conda activate DiffTrip
```

### Install pytorch and cuda toolkit

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### Install other requirements

```
conda install numpy pandas
pip install scikit-learn
```

## Folder Structure

| Folder Name |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|    asset    |              Metadata and preprocessing process              |
|    data     |                   Preprocessed input data                    |
|   results   |             Storage related experimental results             |
|   T-Base    | Transformer-based model (Base) and the Base using clipping-merging strategy (Base-CM) |
| T-Base-WSE  | Transformer-based model using weighted classification loss on start and end points (Weighted Source and Destination with Base, Base-WSE for short) |
|   T-Diff    |                 the source code of DiffTrip                  |
| T-Diff-EPS  |               DiffTrip predicts epsilon (EPS)                |
|  T-Diff-FS  |           DiffTrip using 4 steps fast sampling (FS)          |
|  README.md  |                             --                               |
|   run.bat   |                 Script file for running code                 |

## How to run our programs

The detailed operation mode and parameter settings of each model can be found in **run.bat**. 

```
@echo off

REM Setting Python Interpreter Path
set python_path = Python location of your virtual environment

REM To Run T-Diff(DiffTrip)
REM The DiffTrip set default values T(step) in 32 and beta_t(max_noise) in 0.02, respectively. If you want to change it, try to add --step and --max_noise
python .\T-Diff\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16

python .\T-Diff\train_diffusion.py --dataset Glas --lr 0.01 --batch_size 4 --d_model 64

python .\T-Diff\train_diffusion.py --dataset Edin --lr 0.01 --batch_size 16 --d_model 64

python .\T-Diff\train_diffusion.py --dataset Toro --lr 0.01 --batch_size 8 --d_model 64

REM To Run T-Diff-EPS(predict noise)
python .\T-Diff-EPS\train_diffusion.py --dataset Osak --lr 0.005 --batch_size 4 --d_model 32

python .\T-Diff-EPS\train_diffusion.py --dataset Glas --lr 0.005 --batch_size 4 --d_model 32

python .\T-Diff-EPS\train_diffusion.py --dataset Edin --lr 0.005 --batch_size 16 --d_model 32

python .\T-Diff-EPS\train_diffusion.py --dataset Toro --lr 0.005 --batch_size 8 --d_model 32

REM To Run T-Diff-FS(Fast Sampling)
python .\T-Diff-FS\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16

python .\T-Diff-FS\train_diffusion.py --dataset Glas --lr 0.01 --batch_size 4 --d_model 64

python .\T-Diff-FS\train_diffusion.py --dataset Edin --lr 0.01 --batch_size 16 --d_model 64

python .\T-Diff-FS\train_diffusion.py --dataset Toro --lr 0.01 --batch_size 8 --d_model 64

REM To Run T-Base(Base)
REM max f1(pairs-f1) represent the results of Base-CM and total f1(pairs-f1) represent the results of Base
python .\T-Base\train_base.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 128

python .\T-Base\train_base.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 128

python .\T-Base\train_base.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 128

python .\T-Base\train_base.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 64

REM To Run T-Base-WSE(Weight of Start and End)
python .\T-Base-WSE\train_base.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 128 --se_weight 5

python .\T-Base-WSE\train_base.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 128 --se_weight 5

python .\T-Base-WSE\train_base.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 128 --se_weight 5

python .\T-Base-WSE\train_base.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 64 --se_weight 5
```

If your operating system is **Windows**, you can use the command In the working directory as

```
.\run.bat
```

to run this script file directly.  You can also directly paste commands into the terminal to run the program just like

```
python .\T-Diff\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16
```

% Hope such an implementation could help you on your projects. Any comments and feedback are appreciated.
