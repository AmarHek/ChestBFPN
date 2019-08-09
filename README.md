# ChestBFPN
Bayesian Feature Pyramid Networks for Chest X-ray Segmentation and Automatic Assessment of Cardio-Thoratic Ratio

# Requirements

Python 3.*, keras 2.2.4, keras_contrib, segmentation_models, classification_models, PIL, numpy, pandas, cv2

# Data structure

1) Download [Chest-X-Ray dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) and put it in folder ../input/Chest X-ray-14/img/
2) Download [DeepChest archive]() and put it content in '../input/' folder. It contains: 
- folder '**masks_train**' - train masks
- folder '**masks_test**' - test masks
- file '**train_val_split.pkl**' - split on train and validation parts

# Citation

If you use it please cite: 