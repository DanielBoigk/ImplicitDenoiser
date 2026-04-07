**Still under development, sorry**

Basically I need a denoiser/deblurrer for use in EIT.
So I orient myself around existing architectures like this one:
https://arxiv.org/pdf/2008.13751

Which is not only powerful as a denoiser but also works as a deblurrer if trained for that. One can however also use it to regularize inverse problems in imaging.

However I have a few problems:
-  there are schemes to use networks as regularizer in a Plug and Play(PnP)- manner or ADMM, Split-Bregman, ... for linear inverse problems. However since I have an ill conditioned nonlinear inverse problem and the mathematics gets more complicated than I can handle and it didn't work particularly well, so I would actually prefer to have a gradient to be used in "simple" optimizers like Gauss-Newton, L-BFGS, etc.
- The U-Net structure gives me an unnecessary unfold. If trained inside a [DEQ](https://arxiv.org/pdf/1909.01377) this is actually unnecessary. [SciMLSensitivity](https://docs.sciml.ai/SciMLSensitivity/stable/) delivers me the adjoint rules required to now train such a network.
- Currently I'm using normal convolutions. However I would like to adapt this to arbitrary mesh shapes and arbitrary mesh discretisations not just equidistant quadrilaterals. For that one just needs to replace the inner CNN by a Graph Convolutional network (GCN), Graph Attention Network (GAT) or similar.

Other considerations:
- I'm unsure whether it is better to train with an encoder, or whether to enroll at any time.
- No idea how to handle the boundary nodes. 
- This network is easily made equivariant/invariant towards various affine transformations.
- The training scheme didn't turn out to be that easy. Basically one needs to start with a NeuralODE and train as a denoiser for fixed noise levels, then one goes on with variable gaussian noise in a skip-DEQ. Further I have not pogressed. Maybe I need to edit the loss function to handle spectral noise. Once it can handle spectral noise also it should be usable as a prior for learning arbitrary image corruptions... That was the idea at least. 

Some relevant papers:
[Multiscale Deep Equilibrium Networks](https://arxiv.org/pdf/2006.08656)
[Continuous Deep Equilibrium Models](https://arxiv.org/pdf/2201.12240)



Will post the link to Huggingface and the weights once I'm somewhat fine with the result.
