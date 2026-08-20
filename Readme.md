**Denoiser trained on TinyImageNet**

This is a list of experiments to train a denoiser/deblurrer specifially to act as a regularizer for EIT. 

In the end it turned out to be a Julia implementation of this two paper:
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

The resulting Architechture is a Stochastic differential equation trained as the variance preserving variant of a forward diffusion SDE. As data [TinyImageNet](https://www.kaggle.com/competitions/tiny-imagenet/data) was used.

Sample images generated for the already trained models are: 

Score based Convolutional Net 

![sample 1](notebooks/SBGM_SDE/TinyImg/samples/sample_1_2026-08-20T17:46:27.998.png)
![sample 2](notebooks/SBGM_SDE/TinyImg/samples/sample_2_2026-08-20T17:46:27.998.png)
![sample 3](notebooks/SBGM_SDE/TinyImg/samples/sample_3_2026-08-20T17:46:27.998.png)
![sample 4](notebooks/SBGM_SDE/TinyImg/samples/sample_4_2026-08-20T17:46:27.998.png)
![sample 5](notebooks/SBGM_SDE/TinyImg/samples/sample_5_2026-08-20T17:46:27.998.png)
![sample 6](notebooks/SBGM_SDE/TinyImg/samples/sample_6_2026-08-20T17:46:27.998.png)
![sample 7](notebooks/SBGM_SDE/TinyImg/samples/sample_7_2026-08-20T17:46:27.998.png)
![sample 8](notebooks/SBGM_SDE/TinyImg/samples/sample_8_2026-08-20T17:46:27.998.png)

U-Net

![sample 1](notebooks/UNet/TinyImg/samples/sample_1_2026-08-18T22:09:07.404.png)
![sample 2](notebooks/UNet/TinyImg/samples/sample_2_2026-08-18T22:09:07.404.png)
![sample 3](notebooks/UNet/TinyImg/samples/sample_3_2026-08-18T22:09:07.404.png)
![sample 4](notebooks/UNet/TinyImg/samples/sample_4_2026-08-18T22:09:07.404.png)
![sample 5](notebooks/UNet/TinyImg/samples/sample_5_2026-08-18T22:09:07.404.png)
![sample 6](notebooks/UNet/TinyImg/samples/sample_6_2026-08-18T22:09:07.404.png)
![sample 7](notebooks/UNet/TinyImg/samples/sample_7_2026-08-18T22:09:07.404.png)
![sample 8](notebooks/UNet/TinyImg/samples/sample_8_2026-08-18T22:09:07.404.png)

From hereon we can try things like: 
- [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)
- [RED-Diff](https://arxiv.org/pdf/2305.04391)
- [Diff-PIR](https://arxiv.org/pdf/2305.08995)
- [Opt-Diff](https://arxiv.org/pdf/2605.11506)