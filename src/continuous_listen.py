"""
Continuosly listens for the wake word. 
Sources:
- https://www.youtube.com/watch?v=gl1lhplZmaI&t=1s
- https://docs.python.org/3/library/threading.html
"""

import threading 
import sounddevice as sd 
from scipy.io.wavfile import write
import librosa
import constants as const
import torch 
from train import WakeWord
import time 


dropout = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load saved model 
print("Loading model...")
start_time = time.time()
model = WakeWord(const.NUM_MFCC * const.MFCC_Y, dropout_rate=dropout)
model.load_state_dict(torch.load("./model/ARISE.pth"))
model.eval()
model.to(device)
print("Loaded model in ", time.time() - start_time, "s")


def listen(stop_event): 
    """
    Listens for the wake word and calls the detect function with the mfcc
    """
    print("Listening...")
    while not stop_event.is_set():
        recording = sd.rec(const.SAMPLE_RATE * const.DURATION, 
                        samplerate=const.SAMPLE_RATE, 
                        channels=2) 
        sd.wait()    
        audio_mono = librosa.to_mono(recording.T)
        mfcc = librosa.feature.mfcc(y=audio_mono, sr=const.SAMPLE_RATE, n_mfcc=const.NUM_MFCC)
        detect_thread = threading.Thread(target=detect, args=(mfcc,))
        detect_thread.start()


def detect(mfcc): 
    """
    Detect the wake word in the given mfcc
    """
    pred = model(torch.tensor(mfcc, device=device).unsqueeze(0).float())

    if pred > const.THRESHOLD: 
        print("Wake word detected!")
    else: 
        print("No wake word detected...")
    
if __name__ == "__main__": 
    stop_event = threading.Event()

    listen_thread = threading.Thread(target=listen, args=(stop_event,))
    listen_thread.start()
    
    try: 
        while True: 
            time.sleep(0.01) # main thread does nothing 
    except KeyboardInterrupt:
        print("Stopping all threads...")
        stop_event.set()    

    listen_thread.join()