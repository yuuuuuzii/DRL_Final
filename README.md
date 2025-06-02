# HN-SAC: Incorporating Hyper Networks with SAC for Continual Learning

This repository is the implementation of a hypernetwork-augmented Soft Actor-Critic (SAC) framework for continual task
learning in the HalfCheetah-v4 environment.

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate hypersac
```

## Training

We implement continual learning on three sequential “joint‐failure” tasks in HalfCheetah‐v4. Our main method is Hypernetwork + SAC, which we compare against three benchmarks:
- Single‐Task SAC
- Continual SAC
- EWC + SAC

To train the model(s) in the paper, run this command:
### 1. Hypernetwork + SAC (Our method)
A small Hypernetwork takes a multi‐hot task embedding and outputs actor/critic weights.
```train
python train_hyper.py
```
### 2. Single‐Task SAC
Trains three independent SAC agents — one per task (no sequential transfer).
```train
python train_single.py
```
### 3. Continual SAC
Trains a single SAC agent sequentially without any continual‐learning mechanism.
```train
python train_continual.py
```
### 4. EWC + SAC
Same pipeline as “train_continual,” but adds an Elastic Weight Consolidation penalty to preserve previous‐task parameters.
```train
python train_EWC.py
```

## Evaluation

To evaluate Hypernetwork + SAC against vanilla Continual SAC:
```eval
python eval.py
```
