**Still under development, sorry**

Basically I need a denoiser/deblurrer for use in EIT.
So I orient myself around existing architectures like this one:
https://arxiv.org/pdf/2008.13751

Which is not only powerful as a denoiser but also works as a deblurrer if trained for that. One can however also use it to regularize inverse problems in imaging.

However I have a few problems:
-  there are schemes to use networks as regularizer in a Plug and Play(PnP)- manner or ADMM, Split-Bregman, ... for linear inverse problems. However since I have an ill conditioned nonlinear inverse problem and the mathematics gets more complicated than I can handle and it didn't work particularly well, so I would actually prefer to have a gradient to be used in "simple" optimizers like Gauss-Newton, L-BFGS, etc.
- The U-Net structure gives me an unnecessary unfold. If trained inside a [DEQ](https://arxiv.org/pdf/1909.01377) this is actually unnecessary. [SciMLSensitivity]() delivers me the adjoint rules required to now train sucha a network.
- Currently I'm using normal convolutions. However I would like to adapt this to arbitrary mesh shapes and arbitrary mesh discretisations not just equidistant quadrilaterals. For that one just needs to replace the inner CNN by a Graph Convolutional network (GCN), Graph Attention Network (GAT) or similar.
- This network is easily made equivariant/invariant towards various affine transformations. 
- The training scheme didn't turn out to be that easy. Basically one needs to start with a neuralODE and train as a denoiser for a fixed noise level, then one goes on variable gaussian noise in a skip-DEQ, maybe I need to 

Some relevant papers:
[Multiscale Deep Equilibrium Networks](https://arxiv.org/pdf/2006.08656)
[Continuous Deep Equilibrium Models](https://arxiv.org/pdf/2201.12240)



Will post the link to Huggingface and the weights once I'm somewhat fine with the result.
