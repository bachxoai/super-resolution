## Super Resolution Examples

- Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)



### Prepare Pre-trained VGG and Trained model

#### Pre-trained VGG

- You need to download the pretrained VGG19 model weights in [here](https://drive.google.com/file/d/1CLw6Cn3yNI1N15HyX99_Zy9QnDcgP3q7/view?usp=sharing).
- Put the weights file under the folder `srgan/model/`.

#### Trained model

🔥 Download model weights as follows.
- Download generator and discriminator weights file in [here](https://drive.google.com/drive/folders/1XkYAuF1E-2W05SMC--sipx-FrOHjQz6z?usp=sharing).

- Put weight files under the folder `srgan/models/`.

Your directory structure should look like this:

```
srgan/
    └── config.py
    └── srgan.py
    └── train.py
    └── vgg.py
    └── model
          └── vgg19.npy
    └── models
          ├── g.npz
          └── d.npz

```


### Run

🔥🔥🔥🔥🔥🔥 Please install all the dependencies:

```bash
pip install -r requirements.txt
```
#### Test

- Start testing
```bash
python train.py --mode=test
```

Results will be saved under the folder `test/output/`.

#### Train
- Start training.

```bash
python train.py
```


#### Evaluation.

- Start evaluation.
```bash
python train.py --mode=eval
```

Results will be saved under the folder srgan/samples/. 