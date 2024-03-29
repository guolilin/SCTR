import torchio as tio
from pathlib import Path
import torch
import numpy as np
import copy
from medpy.metric.binary import hd, hd95
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

def do_subject(image_paths, label_paths):
    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        print(image_path)
        print(label_path)
        subject = tio.Subject(
            pred=tio.ScalarImage(image_path),
            gt=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    return subjects

# predict_dir = './results/adam_residual_unet3d/train'
# labels_dir = './dataset/train_label'

predict_dir = './results/adam_vnet/test'
labels_dir = './dataset/test_label'

# predict_dir = './results/adam_residual_unet3d/debug'
# labels_dir = './dataset/debug_label'

images_dir = Path(predict_dir)
labels_dir = Path(labels_dir)

image_paths = sorted(images_dir.glob('*.nii.gz'))
label_paths = sorted(labels_dir.glob('*.nii.gz'))

training_set = do_subject(image_paths, label_paths)
pre = 0.0
rec = 0.0
dsc = 0.0
for i,subj in enumerate(training_set):
    gt = subj['gt'][tio.DATA]
    pred = subj['pred'][tio.DATA]#.permute(0,1,3,2)

    preds = pred.numpy()
    gts = gt.numpy()

    gts[gts == 2] = 0

    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    jaccard = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    print('precision:', precision)
    print('recall:   ', recall)
    print('dice:     ', dice)
    print(tp, fp, tn, fn)
    pre += precision
    rec += recall
    dsc += dice

print('-----------------------------')
print('precision:', pre / len(training_set))
print('recall:   ', rec / len(training_set))
print('dice:     ', dsc / len(training_set))
