#!/bin/bash

# By default, this script runs a single training and testing of Patchcore on the AutoVI engine_wiring class.
# First argument is the method name in [cflow, dsr, draem, efficient_ad, padim, patchcore]
# Second argument is the name of the category in [engine_wiring, pipe_clip, pipe_staple, tank_screw, underbody_pipes, underbody_screw]
# Note that changing the category requires changing the category in the corresponding config_train.yaml file in Anomalib

method=patchcore
category=engine_wiring

if [ -z "$1" ]; then
    echo "Default method selected: patchcore"
else
    method=$1
    echo "Selected method: $method"
fi

if [ -z "$2" ]; then
    echo "Default category selected: engine_wiring"
else
    category=$2
    echo "Selected category: $category"
fi


cd anomalib
python3 tools/train.py --config src/anomalib/models/$method/config_train.yaml

cd ..
mkdir -p results/$method/$category

mv anomalib/results/$method/autovi/$category/run/pretty_heatmaps results/$method/$category/heatmaps/
mv anomalib/results/$method/autovi/$category/run/logs/lightning_logs/*/metrics.csv results/$method/$category/classification.csv
python3 spro/evaluate_experiment.py --object_name $category --dataset_base_dir anomalib/datasets/AutoVI/ --anomaly_maps_dir anomalib/results/$method/autovi/$category/run/images/ --output_dir results/

mv results/metrics.json results/$method/$category/segmentation.json
