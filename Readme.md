**Denoiser trained on TinyImageNet**

This is a list of experiments to train a denoiser/deblurrer specifially to act as a regularizer for EIT. 

In the end it turned out to be a Julia implementation of this two paper:
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

The resulting Architechture is a Stochastic differential equation trained as the variance preserving variant of a forward diffusion SDE. As data [TinyImageNet](https://www.kaggle.com/competitions/tiny-imagenet/data) was used.

Sample images generated for the already trained models are: 

Score based Convolutional Net 

![sample 1](assets/sbgm_sde_sample_1.png)
![sample 2](assets/sbgm_sde_sample_2.png)
![sample 3](assets/sbgm_sde_sample_3.png)
![sample 4](assets/sbgm_sde_sample_4.png)
![sample 5](assets/sbgm_sde_sample_5.png)
![sample 6](assets/sbgm_sde_sample_6.png)
![sample 7](assets/sbgm_sde_sample_7.png)
![sample 8](assets/sbgm_sde_sample_8.png)

U-Net

![sample 1](assets/unet_sample_1.png)
![sample 2](assets/unet_sample_2.png)
![sample 3](assets/unet_sample_3.png)
![sample 4](assets/unet_sample_4.png)
![sample 5](assets/unet_sample_5.png)
![sample 6](assets/unet_sample_6.png)
![sample 7](assets/unet_sample_7.png)
![sample 8](assets/unet_sample_8.png)

From hereon we can try things like: 
- [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)
- [RED-Diff](https://arxiv.org/pdf/2305.04391)
- [Diff-PIR](https://arxiv.org/pdf/2305.08995)
- [Opt-Diff](https://arxiv.org/pdf/2605.11506)