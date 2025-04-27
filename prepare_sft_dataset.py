from data_process.audio_process import DataProcess

def main():
    data_process = DataProcess()
    data_process.pipeline(input_dir="data", output_dir="data")
    
if __name__ == "__main__":
    main()