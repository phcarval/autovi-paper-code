# AutoVI dataset
This repository contains the code for training and evaluating the Automotive Visual Inspection Dataset (AutoVI).

## Source code for evaluation
This source code is largely based on the MVTec LOCO evaluation code available at [https://www.mvtec.com/company/research/datasets/mvtec-loco](https://www.mvtec.com/company/research/datasets/mvtec-loco) and the Anomalib library available at [https://github.com/openvinotoolkit/anomalib](https://github.com/openvinotoolkit/anomalib).

## Install dependencies
Clone the repository locally, then execute the following commands:
```bash
cd paper_code/
yes | conda create -n anomalib_env python=3.10
conda activate anomalib_env
pip install -e anomalib/
```

## Setup
Please download each AutoVI category from the corresponding GitHub release, then extract each folder to `anomalib/datasets/AutoVI/`.

## Launch training
This script will launch a training and testing of Patchcore on the AutoVI `engine_wiring` category:
```bash
./train.sh
```

The first and second arguments respectively set the method and category:
```bash
./train.sh padim underbody_pipes
```
Note that changing the category requires changing the category in the corresponding config_train.yaml file in Anomalib (`anomalib/src/anomalib/models/model\_name/config\_train.yaml`), near the top of the file.

## Results
Results are available in the `results/` folder: classification and segmentation results.
