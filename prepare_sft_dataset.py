from data_process.audio_process import DataProcess

def main():
    data_process = DataProcess()
    data_process.pipeline(librispeech_dir="/data/common/datasets/LibriSpeech/LibriSpeech", input_dir="data", output_dir="data")
    
if __name__ == "__main__":
    main()