import numpy as np
import pymia.data.transformation as tfm


class LabelsToLong(tfm.Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample: dict) -> dict:
        if 'labels' not in sample:
            return sample

        if sample['labels'].dtype is not np.int64:
            sample['labels'] = sample['labels'].astype(np.int64)
        return sample
