import argparse
import sys 
sys.path.append("audioDeepFake")
from audioDeepFake.inference import pred_audio

def process_audio(audio_path):
    # Your audio processing code goes here
    print("Processing audio file at:", audio_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an audio file.')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    args = parser.parse_args()

    audio_path = args.audio_path

    ret = pred_audio(audio_path)

    output = "Real" if ret == 1 else "Fake"
    print(output)