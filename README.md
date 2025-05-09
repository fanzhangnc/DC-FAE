# DC-FAE

Official code for `Dynamic Dual Consistency Distillation for Imbalanced Facial Attribute Editing`.

## Getting Started

### Requirements

The code is tested on `python 3.9, Pytorch =2.0.1`.

```
git clone https://github.com/fanzhangnc/DC-FAE.git
cd DC-FAE/
conda create -n your_env_name python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Pre-trained Models

Additionally, various auxiliary models are provided for training your own model from scratch. These models include [E4E](https://github.com/omertov/encoder4editing) encoder, official StyleGAN2 model, and other utility models.

### E4E Encoder

| Path                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [E4E encoder](https://pan.baidu.com/s/1ewtwq56nmnd1_PVhta-jvA?pwd=geia) | Pretrained e4e encoder is utilized to project facial features into the W+ space. |

### StyleGAN2 Model

| Path                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [StyleGAN trained on the FFHQ](https://pan.baidu.com/s/1ewtwq56nmnd1_PVhta-jvA?pwd=geia) | StyleGAN2 model trained on FFHQ with 1024x1024 output resolution. |

### Other Utility Models

| Path                                                                                                       | Description                                                                                                                                                                                   |
|:-----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ResNet-34 and ResNet-50 Models](https://pan.baidu.com/s/1ewtwq56nmnd1_PVhta-jvA?pwd=geia) | Pretrained ResNet-34 and ResNet-50 models are used as the backbone of our attribute classifier to predict attributes for manipulated faces. |
| [Real NVP Model](https://pan.baidu.com/s/1ewtwq56nmnd1_PVhta-jvA?pwd=geia)                 | A predefined density real NVP model constrains transformed latent codes within the latent space distribution. |
| [MobileFaceNet Model](https://pan.baidu.com/s/1ewtwq56nmnd1_PVhta-jvA?pwd=geia)   | Pretrained MobileFaceNet model was taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) and is for use in our ID loss. |

Download all needed models, and put them into `data/`.

## Preparing Your Data

Download the [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets. The age labels for FFHQ can be obtained from [here](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis/tree/master). CelebA should be aligned similarly to FFHQ; for details, see https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py.

## Training Your Own Models

We provide the training code in `training`.

1. `training/train_realnvp_and_classifier.ipynb` trains the attribute classifier, the Real NVP model, and extracts the latent codes.

2. `training/latent_translation.py` trains teacher models to edit attributes such as gender and eyeglasses. Intermediate training results, including checkpoints and images, are saved to `training/dual_trans_ckpts/opts.run_name`.

Here, we provide example scripts for training on single attributes:
```sh
# Eyeglasses
python latent_translation.py --run_name 15_1221 --keeps 20 -1 --changes 15
# Gender
python latent_translation.py --run_name 20_1221 --keeps 15 -1 --changes 20
# Young
python latent_translation.py --run_name 39_1221 --keeps 15 20 --changes 39
```

### Additional Notes

- You can use the `--changes` flag to define the attribute(s) to be edited by the teacher models.

    - By default, `--changes` is set to `15` to train the teacher models to manipulate eyeglasses.
    - To train the teacher models to edit multiple attributes, pass a list to `--changes`, e.g., `[15, 20]`, to manipulate both eyeglasses and gender simultaneously.
	
- You can define the maximum steps of the transformation trajectory with the `--max_steps` flag, which defaults to `5`.

3. `training/transform_and_save_latents.py` uses the pre-trained teacher models to process training samples, integrate results via teacher selection, and save the intermediate edited latent codes as supervision signals for distillation.

Example scripts:
```sh
# Eyeglasses
python transform_and_save_latents.py --run_name 15_1221 --keeps 20 -1 --changes 15
# Gender
python transform_and_save_latents.py --run_name 20_1221 --keeps 15 -1 --changes 20
# Young
python transform_and_save_latents.py --run_name 39_1221 --keeps 15 20 --changes 39
```

4. `training/train_compressed_model.py` trains a single model using the DC-FAE.

Example scripts:
```sh
# Eyeglasses
python train_compressed_model.py --eta [1,1,1,1,1] --gamma [0.001,0.01,0.05,0.1,1] --run_name 15_1221 --keeps 20 -1 --changes 15 --grl --out_layer [0,1,2,3,4]
# Gender
python train_compressed_model.py --eta [1,1,1,1,1] --gamma [0.001,0.01,0.05,0.1,1] --run_name 20_1221 --keeps 15 -1 --changes 20 --grl --out_layer [0,1,2,3,4]
# Young
python train_compressed_model.py --eta [1,1,1,1,1] --gamma [0.001,0.01,0.05,0.1,1] --run_name 39_1221 --keeps 15 20 --changes 39 --grl --out_layer [0,1,2,3,4]
```

### Additional Notes
	
- To select the steps where intermediate outputs from the teacher selection module guide the student model, you can use the `--out_layer`.

  - It is set to `[-1]` by default, meaning only the final output is used.
  - You can use weighted coefficients with the `--gamma` to balance the contribution of each step in the weighted L2 loss function, which defaults to `[1] * 10`.

## Testing

Make sure that the attribute classifier is downloaded to the `data/` directory and the E4E encoder is prepared as required. After training your own models, you can use `evaluation.ipynb` to evaluate their performance on the test set images.

## Directory Structure

```
DC-FAE
│  .gitignore
│  evaluation.ipynb                         # Use to evaluate model performance
│  README.md
│  requirements.txt                         # List of dependencies
│  
├─data                                      # Directory for pretrained models
├─models
│  │  dataset.py                            # Dataset handling
│  │  decoder.py                            # StyleGAN decoder model
│  │  modules.py
│  │  
│  ├─e4e                                    # e4e code
│  ├─face_align                             # Face alignment utilities
│  ├─face_seg                               # Face segmentation utilities
│  ├─flows                                  # Real NVP model
│  ├─ops                                    # Operations and utilities
│  └─stylegan2                              # StyleGAN2 code
└─training
        latent_translation.py               # Trains teacher models
        train_compressed_model.py           # Trains a single model using the DC-FAE
        train_realnvp_and_classifier.ipynb  # Trains classifier, Real NVP, and extracts latent codes
        transform_and_save_latents.py       # Uses teacher models to process samples and save latents for distillation
```

## Contact

If you have any questions, please feel free to contact fanzhang@email.ncu.edu.cn.

## Acknowledgments

This implementation builds upon the awesome work done by Zhizhong Huang et al. (**[AdaTrans](https://github.com/Hzzone/AdaTrans)**), Karras et al. (**[StyleGAN2](https://github.com/NVlabs/stylegan2)**) and Chen et al. (**[torchdiffeq](https://github.com/rtqichen/torchdiffeq)**). They provides inspiration and examples that played a crucial role in the implements of this project.