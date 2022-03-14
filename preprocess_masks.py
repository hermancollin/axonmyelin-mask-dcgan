"""
This script applies some basic preprocessing to the TEM masks to
prepare the data for training
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
    for subj in subject_dict.keys():
        sample_count = 0
        deriv_path = DATAPATH / 'derivatives' / 'labels' / subj / 'micr'
        masks = list(deriv_path.glob('*axonmyelin*'))
        for m in masks:
            image = ads_utils.imread(str(m))
            # resizing smallest dimension from 2286 to 2048
            image = cv2.resize(image, dsize=(3370, 2048), interpolation=cv2.INTER_CUBIC)
            tile1 = image[:1024, :1024]
            tile2 = image[1024:, :1024]
            tile3 = image[:1024, 1024:2048]
            tile4 = image[1024:, 1024:2048]
            tile5 = image[:1024, 2048:3072]
            tile6 = image[1024:, 2048:3072]
            tiles = [tile1, tile2, tile3, tile4, tile5, tile6]
            for tile in tiles:
                fname = OUTPATH / Path(subj + '_' + str(sample_count))
                sample_count += 1
                ads_utils.imwrite(str(fname), tile)
    

if __name__ == '__main__':
    main()