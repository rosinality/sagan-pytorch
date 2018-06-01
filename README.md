# sagan-pytorch

Self-Attention Generative Adversarial Networks (SAGAN, https://arxiv.org/abs/1805.08318) in PyTorch

Usage:

> python train.py PATH

Input directory should be structured like this (as with torchvision.datasets.ImageFolder):

> PATH/class1 <br/>
> PATH/class2 <br/>
> ...

Code for evaulate FID score came from https://github.com/bioinf-jku/TTUR

## Notes

![Sample](sample.png)

Sample from DCGAN generator (without residual connection) at 120k iteration. Seems like that model size is insufficient. FID is about 120. After this model is collapsed.

Currently trying ResNet generator/discriminator. But it was hard to train ResNet model with generator learning rate 1e-4/discriminator learning rate 4e-4 schedule as in the paper. Maybe smaller learning rate of generator is preferable.