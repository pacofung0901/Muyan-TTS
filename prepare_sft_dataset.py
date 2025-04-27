from data_process.audio_process import DataProcess

def main():
    data_process = DataProcess()
    # Specify your local path to the librispeech data.
    data_process.pipeline(librispeech_dir="path/to/librispeech", input_dir="data", output_dir="data")
    
if __name__ == "__main__":
    main()