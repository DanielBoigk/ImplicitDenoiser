**Denoiser trained on TinyImageNet**

This is a list of experiments to train a denoiser/deblurrer specifially to act as a regularizer for EIT. 

In the end it turned out to be a Julia implementation of these two papers:
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)

The resulting Architechture is a Stochastic differential equation trained as the variance preserving variant of a forward diffusion SDE. As data [TinyImageNet](https://www.kaggle.com/competitions/tiny-imagenet/data) was used