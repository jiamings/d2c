
## Training scripts

Our training currently involves two steps. First step is training the auto-encoder with contrastive loss. Once, trained the feature vectors are extracted from the encoder over the training set. DDIM model is then trained over the latent space.

### Training over CIFAR-10/CIFAR-100 datasets
Run the following script.
```
bash train_cifar.sh
```
The above script will train the Auto-Encoder first, followed by the training of DDIM over the latent space. It generates 3 ckpts, one for AutoEncoder, one for DDIM and one containing some stats about latent space.

### Converting the ckpts
To combine the 3 ckpts such that it can be used for sampling and editing, run the following for example
```
python convert_ckpt.py --AE_ckpt C10_ckpts/checkpoint_recent.pth.tar --DDIM_ckpt ../d2c/diffusion/cifar10_model/logs/run_0/ckpt.pth --latent_ckpt C10_ckpts/latent_stats.ckpt --output_path combined_C10.ckpt
```

## Todo

- [ ] Release training code for FFHQ


## References and Acknowledgements

If you find this repository useful for your research, please cite our work.
```
@inproceedings{sinha2021d2c,
  title={D2C: Diffusion-Denoising Models for Few-shot Conditional Generation},
  author={Sinha*, Abhishek and Song*, Jiaming and Meng, Chenlin and Ermon, Stefano},
  year={2021},
  month={December},
  abbr={NeurIPS 2021},
  url={https://arxiv.org/abs/2106.06819},
  booktitle={Neural Information Processing Systems},
  html={https://d2c-model.github.io}
}
```

This implementation is based on:
- [DDIM](https://github.com/ermongroup/ddim).
- [NVAE](https://github.com/nvlabs/nvae).
