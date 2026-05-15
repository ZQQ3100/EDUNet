# EDUNet

Event-Guided Deep Unfolding for Lightweight and Interpretable Motion Deblurring.

This work is currently submitted to The Visual Computer.

## Dependencies

- Python 3.6
- PyTorch 1.10.2 (CUDA 11.3)
- torchvision 0.11.3
- CUDA 11.3 (for GPU support)

You can create a new [Anaconda](https://www.anaconda.com/products/individual) environment with the above dependencies as follows.
Please make sure to adapt the CUDA toolkit version according to your setup when installing torch and torchvision.

All required packages are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

## Datasets

There are three kinds of data:

- synthetic GoPro dataset (**gopro_test**) from [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html) and [ESIM](http://rpg.ifi.uzh.ch/esim.html),
- REVD dataset (**HQF_test**) from [REVD](https://sites.google.com/view/fevd-cvpr2024)
- real-world scenes (RWS) dataset (**realdata_test**) from [realdata_test](https://drive.google.com/drive/folders/1ODMevq1aeVuIXCiDpSzEbaJ6cZNowIEe?usp=sharing).

## Quick start

#### Initialization

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

If you find this work helpful for your research, please consider citing our paper (to be published):
```
@article{yu2022learning,
  title={Event-Guided Deep Unfolding for Lightweight and Interpretable Motion Deblurring},
  author={Qi, Na and Zhao, Qianqian and Yue, Huanjing and Chen, Liang},
  booktitle={The Visual Computer}
  year={2026},
}
```
If you use the code, please cite this code repository:

[![DOI](https://zenodo.org/badge/1222664892.svg)](https://doi.org/10.5281/zenodo.20213365)

## Related Projects
[esL-Net++](https://github.com/ShinyWang33/eSL-Net-Plusplus)