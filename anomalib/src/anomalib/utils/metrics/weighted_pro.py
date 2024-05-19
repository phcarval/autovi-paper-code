"""Implementation of weighted PRO metric based on TorchMetrics."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional import binary_jaccard_index
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.metrics import PRO


class WeightedPRO(PRO):
    """Weighted Per-Region Overlap (weighted PRO) Score."""

    def compute(self) -> Tensor:
        """Compute the macro average of the weighted PRO score across all regions in all batches."""
        pro = super().compute()

        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)

        target = target.unsqueeze(1).type(torch.float)  # kornia expects N1HW and FloatTensor format
        
        predictions = 1-preds
        weighted_pro = pro - recall(predictions.flatten(), target.flatten(), threshold = self.threshold)

        return weighted_pro

