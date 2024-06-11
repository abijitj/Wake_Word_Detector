"""
File that processing audio samples and saves them as MFCCs to be fed into a 
wake word detection model. 
Sources: 
- https://www.youtube.com/watch?v=NITIefkRae0&t
- https://librosa.org/doc/latest/index.html
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Save file paths of train wav files 
def get_file_paths(path):
    """
    Iterate through given directory path and save file paths of all wav files. 
    """
    file_paths = []
    for dir in os.listdir(path):
        if dir.endswith('.wav'):
            file_paths.append(path + dir)
    return file_paths

file_paths = {
    0: get_file_paths('./Data/train/neg/'),
    1: get_file_paths('./Data/train/pos/'),
}

# Convert each file to MFCC 
def convert_to_mfcc(file_paths):
    """
    Convert each file to MFCC and save as a numpy array. 
    """
    mfccs = []
    for file in file_paths:
        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs.append(mfcc)
    return np.array(mfccs)

mfccs = {
    0: convert_to_mfcc(file_paths[0]),
    1: convert_to_mfcc(file_paths[1]), 
}

all_data = [[mfcc, label] for label, mfccs in mfccs.items() for mfcc in mfccs]
pd.DataFrame(all_data, columns=['mfcc', 'label']).to_pickle('mfccs.csv')





