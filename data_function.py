from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import math
import random
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
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
from pathlib import Path

from hparam import hparams as hp


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, locs_dir):

        if hp.mode == '3d':
            patch_size = hp.patch_size
        elif hp.mode == '2d':
            patch_size = hp.patch_size
        else:
            raise Exception('no such kind of mode!')

        self.subjects = []
        znorm = ZNormalization()

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))
            locs_dir = Path(locs_dir)
            self.loc_paths = sorted(locs_dir.glob('*.txt'))

            sum_pos = 0
            sum_neg = 0
            for (image_path, label_path, loc_path) in zip(self.image_paths, self.label_paths, self.loc_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                # print(image_path)
                subject = znorm(subject)
                x, y, z, rad = [], [], [], []
                with open(loc_path) as loc:
                    for line in loc:
                        cord = line.split()
                        x.append(int(cord[0].split(',')[0]))
                        y.append(int(cord[1].split(',')[0]))
                        z.append(int(cord[2].split(',')[0]))
                        rad.append(math.ceil(float(cord[3])) * 2)
                grid_sampler = torchio.inference.GridSampler(
                        subject,
                        hp.patch_size,
                        hp.patch_overlap,
                )
                pos = 0
                neg = 0
                pos_sample = 0
                neg_sample = 0
                for patch in grid_sampler:
                    for i in range(len(x)):
                        if patch[torchio.LOCATION][0] < x[i] - rad[i] and patch[torchio.LOCATION][3] > x[i] + rad[i]:
                            if patch[torchio.LOCATION][1] < y[i] - rad[i] and patch[torchio.LOCATION][4] > y[i] + rad[i]:
                                if patch[torchio.LOCATION][2] < z[i] - rad[i] and patch[torchio.LOCATION][5] > z[i] + rad[i]:
                                    #print(patch[torchio.LOCATION])
                                    pos += 1
                                    break
                nsd = int((len(grid_sampler) - pos) / 200)
                for patch in grid_sampler:
                    hit = 0
                    for i in range(len(x)):
                        if patch[torchio.LOCATION][0] < x[i] - rad[i] and patch[torchio.LOCATION][3] > x[i] + rad[i]:
                            if patch[torchio.LOCATION][1] < y[i] - rad[i] and patch[torchio.LOCATION][4] > y[i] + rad[i]:
                                if patch[torchio.LOCATION][2] < z[i] - rad[i] and patch[torchio.LOCATION][5] > z[i] + rad[i]:
                                    hit = 1
                                    break
                    if hit == 1:
                        for i in range(int(100/pos) + 1):
                            pos_sample += 1
                            self.subjects.append(patch)
                    else:
                        if nsd == 0 or neg % (nsd + 1) == 0:
                            self.subjects.append(patch)
                            neg_sample += 1
                        neg += 1
                # print('pos_sample: ', pos_sample)
                # print('neg_sample: ', neg_sample)
                sum_pos += pos_sample
                sum_neg += neg_sample
                # if sum_pos > 0 and sum_neg > 0:
                #     break
            print(sum_pos, sum_neg)

        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/artery')
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.fold_arch))

            lung_labels_dir = Path(labels_dir+'/lung')
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.fold_arch))

            trachea_labels_dir = Path(labels_dir+'/trachea')
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.fold_arch))

            vein_labels_dir = Path(labels_dir+'/vein')
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.fold_arch))


            for (image_path, artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths,self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    atery=tio.LabelMap(artery_label_path),
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )
                self.subjects.append(subject)


        #self.transforms = self.transform()

        #self.training_set = tio.SubjectsDataset(self.subjects, transform=None)



    def transform(self):

        if hp.mode == '3d':
            if hp.aug:
                training_transform = Compose([
                # ToCanonical(),
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')


        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):


        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))
            idx = 0
            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                #if idx >= 20:
                #    break
                #if str(label_path).split('/')[-1] != 'label_10060F.nii.gz':
                #    continue
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append([subject, label_path])
                idx += 1




