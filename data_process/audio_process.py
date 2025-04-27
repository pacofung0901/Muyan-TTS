from data_process import audio_hubert_embedding, audio_tokenizer, text_audio_combine, text_format_conversion
import os

class DataProcess():
    def __init__(self):
        # 这个类主要是用来调用训练数据组织的代码的
        self.text_format_conversion = text_format_conversion.generate_raw_data_list
        self.audio_hubert_embedding = audio_hubert_embedding.audio_hubert_embedding
        self.audio_tokenizer = audio_tokenizer.audio_tokenizer
        self.text_audio_combine = text_audio_combine.text_audio_combine
        
    def pipeline(self, input_dir="data", output_dir="data"):
        # 0-用librispeech的数据做例子
        print("step 1: text_format_conversion")
        self.text_format_conversion(librispeech_dir="/data/common/datasets/LibriSpeech/LibriSpeech/", output_dir=output_dir, subset="dev-clean")
        # 2-get_hubert
        print("step 2: audio_hubert_embedding")
        self.audio_hubert_embedding(input_dir=input_dir, output_dir=output_dir)
        # 3-get_semantic
        print("step 3: audio_tokenizer")
        self.audio_tokenizer(input_dir=input_dir, output_dir=output_dir, sovits_path="pretrained_models/Muyan-TTS/sovits.pth")
        print("step 4: text_audio_combine")
        self.text_audio_combine(input_dir=input_dir, output_dir=output_dir)
        print(f"finished process, saved at: {os.path.join(input_dir, 'tts_sft_data.json')}")
        
        
