# Visual Object Networks
[Project Page](http://people.csail.mit.edu/junyanz/projects/von/) |  [Paper](http://people.csail.mit.edu/junyanz/projects/von/VON.pdf)


We present Visual Object Networks (VON), an end-to-end adversarial learning framework that jointly models 3D shapes and 2D images.  Our model can synthesize a 3D shape, its intermediate 2.5D depth representation, and a final 2D image all at once. The VON not only generates images that are more realistic than state-of-the-art 2D image synthesis methods, but also enables several 3D operations.

Note: Code coming tomorrow. 

<img src='imgs/teaser.jpg' width=800>

Visual Object Networks: Image Generation with Disentangled 3D Representation.<br/>
[Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/),
 [Zhoutong Zhang](https://www.csail.mit.edu/person/zhoutong-zhang), [Chengkai Zhang](https://scholar.google.com/citations?user=rChGGwgAAAAJ&hl=en), [Jiajun Wu](https://jiajunwu.com/), [Antonio Torralba](http://web.mit.edu/torralba/www/), [    Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html), [William T. Freeman](http://billf.mit.edu/).<br/>
MIT CSAIL and Google Research.<br/>
In NeurIPS 2018.

## Example results
(a) Typical examples produced by a recent GAN model [Gulrajani et al., 2017].
(b) Our model produces three outputs: a 3D shape, its 2.5D projection given a viewpoint, and a final image with realistic texture.
(c) Given this disentangled 3D representation, our method allows several 3D applications including changing viewpoint and editing shape or texture independently. Please see our code and website for more details.

<img src='imgs/overview.jpg' width=800>

## More Samples
Below we show more samples from DCGAN [Radford et al., 2016], LSGAN [Mao et al., 2017], WGAN-GP [Gulrajani et al., 2017], and our VON. For our method, we show both 3D shapes and 2D images. The learned 3D prior helps our model produce better samples.

<img src='imgs/samples.jpg' width=820>

## 3D Object Manipulations
Our Visual Object Networks (VON) allow several 3D applications such as (left) changing the viewpoint, texture, or shape independently, and (right) interpolating between two objects in shape space, texture space, or both.

<img src='imgs/app.jpg' width=820>

## Texture Transfer across Objects and Viewpoints
VON can transfer the texture of a real image to different shapes and viewpoints

<img src='imgs/transfer.jpg' width=820>

## Prerequisites
- Linux (only tested on Ubuntu 16.04)
- Python3 (only tested with python 3.6)
- Anaconda3


## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/junyanz/VON.git
cd VON
```
-  Install PyTorch 0.4.1+ and torchvision from http://pytorch.org and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by the following:
```bash
conda create --name von --file pkg_specs.txt
source activate von
```

- (Optional) Install [blender](https://www.blender.org/) for visualizing generated 3D shapes. After installation, please add blender to your PATH environment variable.

### Generate 3D shapes, 2.5D sketches, and images
- Download our pretrained models:
```bash
bash ./datasets/download_model.sh
```

-generate results with the model
```
bash ./scripts/figs.sh car
```

The test results will be saved to a html file here: `./results//val/index.html`.

### Model Training
- To train a model, download the training dataset(distance functions and images).
```bash
bash ./datasets/download_dataset.sh
```

- Training 3D generative model:
```bash
bash ./scripts/train_shapes.sh
```
- Training 2D image generation using ShapeNet objects:
```bash
bash ./scripts/train_stage2_real.sh
```

- Train 2D image generation models using trained 3D generator:
```bash
bash ./scripts/train_stage2.sh
```

- Jointly finetune 3D and 2D generative models:
```bash
bash ./scripts/train_full.sh
```

- To view training results and loss plots, go to http://localhost:8097 in a web browser. To see more intermediate results, check out  `./checkpoints/*/web/index.html`


### Citation

If you find this useful for your research, please cite the following paper.
```
@inproceedings{VON,
  title={Visual Object Networks: Image Generation with Disentangled 3{D} Representations},
  author={Jun-Yan Zhu and Zhoutong Zhang and Chengkai Zhang and Jiajun Wu and Antonio Torralba and Joshua B. Tenenbaum and William T. Freeman},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2018}
}

```

### Acknowledgements
This work is supported by NSF #1231216, NSF #1524817, ONR MURI N00014-16-1-2007, Toyota Research Institute, Shell, and Facebook. We thank Xiuming Zhang, Richard Zhang, David Bau, and Zhuang Liu for valuable discussions. This code borrows from the [CycleGAN & pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo.
