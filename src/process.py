"""
File that processes audio samples and saves them as MFCCs to be fed into a 
wake word detection model. 
Sources: 
- https://www.youtube.com/watch?v=NITIefkRae0&t
- https://librosa.org/doc/latest/index.html
"""

import os
import librosa
import numpy as np
import pandas as pd
import constants as const
import sys

if len(sys.argv) < 2: 
    print('Please provide the wake word to process: "arise" or "awaken"')
    exit()

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

if sys.argv[1] == 'arise':
    train_file_paths = {
        0: get_file_paths('./Data/train/neg/'),
        1: get_file_paths('./Data/train/pos-arise/'),
    }

    test_file_paths = {
        0: get_file_paths('./Data/test/neg/'),
        1: get_file_paths('./Data/test/pos-arise/'),
    }
elif sys.argv[1] == 'awaken':
    train_file_paths = {
        0: get_file_paths('./Data/train/neg/'),
        1: get_file_paths('./Data/train/pos-awaken/'),
    }

    test_file_paths = {
        0: get_file_paths('./Data/test/neg/'),
        1: get_file_paths('./Data/test/pos-awaken/'),
    }
else: 
    print('Please provide a valid wake word: "arise" or "awaken"')
    exit()

print("File paths found...")

# Convert each file to MFCC 
def convert_to_mfcc(file_paths):
    """
    Convert each file to MFCC and save as a numpy array. 
    """
    mfccs = []
    for file in file_paths:
        y, sr = librosa.load(file, sr=const.SAMPLE_RATE) # loads as mono channel by default
        # print("y", y.shape, sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=const.NUM_MFCC)
        # print("mfcc", mfcc.shape) # (20, 173)
        mfcc = np.mean(mfcc.T, axis=0) # average the MFCCs over time
        mfccs.append(mfcc)
    return np.array(mfccs)


train_mfccs = {
    0: convert_to_mfcc(train_file_paths[0]),
    1: convert_to_mfcc(train_file_paths[1]), 
}

print("Train MFCCs calculated...")

test_mfccs = {
    0: convert_to_mfcc(test_file_paths[0]),
    1: convert_to_mfcc(test_file_paths[1]), 
}


print("Test MFCCs calculated...")

train_data = []
for label, mfccs in train_mfccs.items():
    for mfcc in mfccs: 
        train_data.append([mfcc, label])

pd.DataFrame(train_data, columns=['mfcc', 'label']).to_pickle('./Data/train/train_mfccs.csv')
# pd.DataFrame(train_data, columns=['mfcc', 'label']).to_csv('./Data/train/train_mfccs2.csv')

test_data = []
for label, mfccs in test_mfccs.items():
    for mfcc in mfccs: 
        test_data.append([mfcc, label])

pd.DataFrame(test_data, columns=['mfcc', 'label']).to_pickle('./Data/test/test_mfccs.csv')

print("CSVs saved...")


