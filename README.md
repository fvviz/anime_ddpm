<h1>Anime DDPM </h1>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>

A Pytorch DDP model trained to generate [Anime faces](https://www.kaggle.com/datasets/splcher/animefacedataset)



<h3> Training results from various epochs: </h3>

![github readme imge](https://github.com/fvviz/anime_ddpm/assets/50950705/feefbe23-c861-4b21-8579-89001d8f8456)

<h3> Denoising visualisation</h3>

https://github.com/fvviz/anime_ddpm/assets/50950705/4042f02f-a5ee-4155-b67c-2b0731e19e42

<h3> Todo </h3>

- Continue training from epoch 69 (i hit the gpu limit on kaggle and colab)
- Fix the cosine scheduler sampler
- Implement EMA from the improved diffusion paper


<h3> References </h3>

- [DDPM paper](https://arxiv.org/pdf/2006.11239.pdf)

- [Improved diffusion](https://arxiv.org/pdf/2102.09672.pdf)

- [Improved diffusion openai codebase](https://github.com/openai/improved-diffusion)

- [Dome272's implementation](https://github.com/dome272/Diffusion-Models-pytorch)




