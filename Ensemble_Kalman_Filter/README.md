# ESE650 Final Project - Ensemble Filtering -- University of Pennsylvania
## Authors
Nicholas Gurnard, Aadit Patel, Pranav Shah

## Requirements and Installation
pytorch, numpy, matplotlib, sklearn, tqdm, click, scipy, cv2 (opencv-python), PyYAML
```
pip install torch
pip install numpy
pip install matplotlib
pip install sklearn
pip install tqdm
pip install click
pip install scipy
pip install opencv-python
pip install PyYAML
```

It is required that you first download the EuRoC MAV datasets in order to run the filters. [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
The data must be placed in the directory `./data/euroc_mav_dataset/`. The files within this dataset must be named `MH_01_easy`, `MH_02_easy`, `MH_03_medium`, `MH_04_difficult`, `MH_05_difficult` in correspondence to the EuRoC dataset naming conventions.

## Runing the code
There are 2 scripts to be run from the root: `train.py` and `test_and_plot.py`.
These 2 scripts are the main part of the project.
Running the flag --help will give instruction on how to run the code with the given datasets and which models to run. Example:

`test_and_plot.py`:
```
python test_and_plot.py --dataset=3 --ensemble=False
```

`train.py`:
```
python train.py --dense=False
```

In order to generate the datasets to run the above scripts, you can run the filters individually. The files
are already provided, but in case you want to run them on new datasets you must run the following:

UKF:
```
python ukf/estimate_rot.py
```
In order to run different datasets, change line 18 `dirname` to point to the correct data

ESKF:
```
python eskf/sandbox_vio.py
```
In order to run different datasets, change line 21 `dirname` to point to the correct data

CF:
```
python complimentary/sandbox_complementary_filter.py
```
In order to run different datasets, change line 16 `fname` to point to the correct data

## Contributions
UKF - Aadit Patel
ESKF - Pranav Shah
CF - Nicholas Gurnard
Ensemble - equal Contributions
Report- equal Contributions
