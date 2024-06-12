"""
File that listens for the specified wake word
Sources: 
- https://librosa.org/doc/main/generated/librosa.load.html
- https://www.youtube.com/watch?v=gl1lhplZmaI&t=1s
"""

import sounddevice as sd 
from scipy.io.wavfile import write
import librosa
import librosa.display
import numpy as np
import constants as const
import torch 
from train import WakeWord
import time 


dropout = 0.5

listening_file = "./listening.wav"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load saved model 
print("Loading model...")
start_time = time.time()
model = WakeWord(const.NUM_MFCC * const.MFCC_Y, dropout_rate=dropout)
model.load_state_dict(torch.load("./model/ARISE.pth"))
model.eval()
model.to(device)
print("Loaded model in ", time.time() - start_time, "s")

while True: 
    print("Listening...")
    recording = sd.rec(const.SAMPLE_RATE * const.DURATION, 
                       samplerate=const.SAMPLE_RATE, 
                       channels=2) 
    # print(recording.shape)
    sd.wait()    
    audio_mono = librosa.to_mono(recording.T)
    # print(audio_mono.shape)
    mfcc = librosa.feature.mfcc(y=audio_mono, sr=const.SAMPLE_RATE, n_mfcc=const.NUM_MFCC)
    # print(mfcc.shape)

    pred = model(torch.tensor(mfcc, device=device).unsqueeze(0).float())

    if pred > const.THRESHOLD: 
        print("Wake word detected!")
    else: 
        print("No wake word detected...")