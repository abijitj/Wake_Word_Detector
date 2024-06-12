""""
File to record audio samples using the sounddevice library. The script can be 
run in two modes: manual and auto.
Sources: 
- https://www.youtube.com/watch?v=NITIefkRae0&t=6s
- https://realpython.com/playing-and-recording-sound-python/

"""

import sounddevice as sd 
from scipy.io.wavfile import write
import sys 

def manual_record(saved_directory, start_count=0, num_samples=32, duration=2, sample_rate=44100, channels=2):
    input("Press Enter to start recording...")
    
    for i in range(start_count, start_count + num_samples):
        recording = sd.rec(sample_rate * duration, samplerate=sample_rate, channels=channels)
        sd.wait()
        write(saved_directory + str(i) + ".wav", sample_rate, recording)
        input("Press Enter to record next sample...")

def auto_record(saved_directory, start_count=0, num_samples=200, duration=2, sample_rate=44100, channels=2):
    print("Starting background recording...")
    count = 1
    for i in range(start_count, start_count + num_samples):
        recording = sd.rec(sample_rate * duration, samplerate=sample_rate, channels=channels)
        # print(recording.shape) # (88200, 2)
        sd.wait()
        write(saved_directory + str(i) + ".wav", sample_rate, recording)
        print(f"Sample {saved_directory + str(i)}.wav recorded...")
    print("Background recordings are complete. Exiting...")

if sys.argv[1] == "manual":
    manual_record(sys.argv[2], start_count=int(sys.argv[3]))
elif sys.argv[1] == "auto":
    auto_record(sys.argv[2], start_count=int(sys.argv[3]))
else: 
    print("Invalid mode. Please select either 'manual' or 'auto'")
