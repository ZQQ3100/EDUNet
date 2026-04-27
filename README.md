# EDUNet

EDUNet: Event-Guided Deep Unfolding Network for Motion Deblurring
This work is currently submitted to The Visual Computer.[Paper]() | [Bibtex]()

## Dependencies

- pyton 3.8
- pytorch >=1.0.0
- torchvision
- argparse
- numpy
- opencv-python
- scipy

You can create a new [Anaconda](https://www.anaconda.com/products/individual) environment with the above dependencies as follows.
Please make sure to adapt the CUDA toolkit version according to your setup when installing torch and torchvision.

```
conda create -n edunet python=3.8
conda activate edunet
conda install pytorch torchvision cudatoolkit -c pytorch
pip install argparse numpy opencv-python scipy 
```

## Datasets

There are three kinds of data:

- synthetic GoPro dataset (**gopro_test**) from [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) and [ESIM](http://rpg.ifi.uzh.ch/esim.html),
- REVD dataset (**HQF_test**) from [REVD](https://sites.google.com/view/fevd-cvpr2024)
- real-world scenes (RWS) dataset (**realdata_test**) from [realdata_test](https://drive.google.com/drive/folders/1ODMevq1aeVuIXCiDpSzEbaJ6cZNowIEe?usp=sharing).

## Quick start

#### Initialization

- Create directory for training data

  `mkdir train_data`

  copy the testing data to directory './train_data/'

- Create directory for testing data

  `mkdir test_data`

  copy the testing data to directory './test_data/'

- Create directory for pretrained model

  `mkdir pre_trained`

  copy the pretrained model to directory './pre_trained/'

#### Training
* To start the training process with the default settings:

```shell
python train.py
```

#### Testing

```shell
python test.py
```

#### Main parameters

- `--load_G`: Path of the pretrained model.
- `--dataset_mode`: The mode of loading dataset, including: `gopro`, `realdata`, `revd`
- `--cuda` : If you use GPU to test, please activate the parameter.

#### Input parameters

- `--input_blur_path` : Path of the input blurry images folder.
- `--input_event_path` : Path of the input event stream folder.

#### Output parameters

- `--output_dir` : Path of the output folder.

## Citation
If you find this work helpful for your research, please consider citing our paper:
```latex

```

## Related Projects
[esL-Net++](https://github.com/ShinyWang33/eSL-Net-Plusplus)