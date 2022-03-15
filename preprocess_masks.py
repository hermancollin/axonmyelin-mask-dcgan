"""
This script applies some basic preprocessing to the TEM masks to
prepare the data for training. Resizes masks to 2048x3370, then 
crops 1024x1024 ROIs
"""

from AxonDeepSeg import ads_utils
from pathlib import Path
import numpy as np
import pandas as pd
import json
import os
import shutil
import cv2


DATAPATH = Path("/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_tem")
OUTPATH = Path("tem_masks") / 'masks'

def main():
    samples_path = DATAPATH / 'samples.tsv'
    samples = pd.read_csv(samples_path, delimiter='\t')
    subject_dict = {}
    for i, row in samples.iterrows():
        subject = row['participant_id']
        sample = row['sample_id']
        if subject not in subject_dict:
            subject_dict[subject] = []
        subject_dict[subject].append(sample)


    os.makedirs(OUTPATH)
    tile_size = 512

    for subj in subject_dict.keys():
        sample_count = 0
        deriv_path = DATAPATH / 'derivatives' / 'labels' / subj / 'micr'
        masks = list(deriv_path.glob('*axonmyelin*'))
        for m in masks:
            image = ads_utils.imread(str(m))
            # resizing smallest dimension from 2286 to 2048
            image = cv2.resize(image, dsize=(3370, 2048), interpolation=cv2.INTER_CUBIC)
            n_xtiles = image.shape[0] // tile_size
            n_ytiles = image.shape[1] // tile_size
            tiles = []
            for i in range(n_xtiles):
                for j in range(n_ytiles):
                    tile = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]
                    tiles.append(tile)
            for tile in tiles:
                fname = OUTPATH / Path(subj + '_' + str(sample_count)+ '.png')
                sample_count += 1
                ads_utils.imwrite(str(fname), tile)
    

if __name__ == '__main__':
    main()