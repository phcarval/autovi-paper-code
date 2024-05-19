from torch import load
from torchmetrics import BinaryROC
import numpy as np
import os
import cv2

def get_roc(ckpt_path, test_path):
    ckpt = os.path.join(ckpt_path, "weights/lightning/model-v1.ckpt")
    model = torch.load(ckpt)
    preds = []
    target = []
    for f in os.listdir(test_path):
        newTarget = None
        if f == "good":
            newTarget = 0
        elif os.path.isdir(os.path.join(test_path, f)):
            newTarget = 1
        else:
            continue
        for i in os.listdir(os.path.join(test_path, f)):
            img = cv2.imread(os.path.join(test_path, f, i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred.append(model(image))
            target.append(newTarget)

    metric = BinaryROC(thresholds = 5000)
    res = metric(preds, target)
    print(res)

get_roc(
    "/home/philippe/metro-calcul/jean-zay/results/heatmaps/patchcore/autovi/v1.0.0/engine_wiring/0",
    "/home/philippe/metro-data/v1.0.0/engine_wiring/test"
)

