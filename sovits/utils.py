import os,re
import traceback
import librosa
import numpy as np
import torch
import ffmpeg
import threading
import subprocess
from sovits.module.mel_processing import spectrogram_torch
from sovits.text.cleaner import clean_text
from sovits.text import cleaned_text_to_sequence
import soundfile as sf
from io import BytesIO
from sovits.LangSegment import LangSegment
import json

def clean_text_inf_normed_text(text, language, version):
    language = language.replace('all_', '')
    _, _, norm_text = clean_text(text, language, version)
    return norm_text

def clean_text_inf_phone(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_normed_text(text,language,version,final=False):
    text = text.replace("&", "and")
    text = text.replace("%", " percent")

    formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    formattext = re.sub(r'\s+([,.!?;:"-])', r'\1', formattext)
    
    norm_text = clean_text_inf_normed_text(formattext, language, version)
    norm_text = norm_text.replace(" ' ", "'")
    # norm_text = re.sub(r"\s(?=[^']*')", '', norm_text)
    return norm_text


def tensor_to_audio_tokens(tensor):
    """
    将一个 Torch Tensor 转换为字符串形式 <|audio_token_x|> 的序列。
    :param tensor: Torch Tensor, 包含整数序列
    :return: 字符串形式的音频 token 序列
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("输入必须是一个 Torch Tensor")
    # 确保 Tensor 是一维
    tensor = tensor.flatten()
    # 将每个整数转换为 <|audio_token_x|> 的形式
    tokens = [f"<|audio_token_{int(val)}|>" for val in tensor.tolist()]
    # 用空格连接 tokens
    result = "".join(tokens)
    return result

def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text

def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)
    return audio_bytes, audio_chunk

def pack_audio(audio_bytes, data, rate, media_type="wav", is_int32=False):
    if media_type == "ogg":
        audio_bytes = pack_ogg(audio_bytes, data, rate)
    elif media_type == "aac":
        audio_bytes = pack_aac(audio_bytes, data, rate, is_int32)
    else:
        # wav无法流式, 先暂存raw
        audio_bytes = pack_raw(audio_bytes, data, rate)
    return audio_bytes

def pack_ogg(audio_bytes, data, rate):
    def handle_pack_ogg():
        with sf.SoundFile(audio_bytes, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
            audio_file.write(data)
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")
    return audio_bytes

def pack_raw(audio_bytes, data, rate):
    audio_bytes.write(data.tobytes())
    return audio_bytes

def pack_wav(audio_bytes, rate, is_int32=False):
    if is_int32:
        data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int32)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV', subtype='PCM_32')
    else:
        data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int16)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV')
    return wav_bytes

def pack_aac(audio_bytes, data, rate, is_int32=False):
    if is_int32:
        pcm = 's32le'
        bit_rate = '256k'
    else:
        pcm = 's16le'
        bit_rate = '128k'
    process = subprocess.Popen([
        'ffmpeg',
        '-f', pcm,  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', bit_rate,  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    audio_bytes.write(out)
    return audio_bytes

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

def should_skip_period(text, dot_index):
    """
    判断在 text[dot_index] == '.' 处，是否需要跳过这个句号。
    规则：
      1) 若句号前只有一个字母(从上一个空格到句号处仅一个字母)，则跳过。
      2) 若句号前单词是 Mr / mr / Dr / dr / Mrs / mrs / st 等，需要跳过。
    """
    # 常见需要跳过的缩写（可自行扩展或改成小写判断）
    skip_abbrs = {"mr", "dr", "mrs", "st"}  # 这里也可以加 "ms", "jr" 等

    # 找到句号前最后一个空格（或开头）
    last_space_index = text.rfind(' ', 0, dot_index)
    if last_space_index == -1:
        start_idx = 0
    else:
        start_idx = last_space_index + 1

    # 取出这个单词（从空格后开始到句号前）
    word_before_dot = text[start_idx:dot_index].strip()
    word_lower = word_before_dot.lower()

    # 1) 若只有一个字母，跳过
    if len(word_lower) == 1:
        return True

    # 2) 若在需要跳过的缩写里，跳过
    if word_lower in skip_abbrs:
        return True

    return False

def fix_spaced_caps(text):
    words = text.split()
    result = []
    i = 0
    
    while i < len(words):
        current_word = words[i]
        # 检查是否是单个大写字母
        if len(current_word) == 1 and current_word.isupper():
            combined = current_word
            j = i + 1
            # 检查后续单词，合并连续的单个大写字母
            while j < len(words):
                next_word = words[j]
                # 如果是单个大写字母，继续合并
                if len(next_word) == 1 and next_word.isupper():
                    combined += next_word
                    j += 1
                # 如果下一个单词以大写字母开头且有后缀（如 's），合并首字母并保留后缀
                elif next_word[0].isupper() and len(next_word) > 1 and not next_word[1:].isalnum():
                    combined += next_word[0]  # 只取首字母
                    combined += next_word[1:]  # 附加剩余部分
                    j += 1
                    break
                else:
                    break
            # 如果合并了多个单词，添加合并结果
            if j > i + 1:
                result.append(combined)
                i = j
            else:
                # 如果只有一个大写字母，且后面不满足合并条件，单独添加
                result.append(current_word)
                i += 1
        else:
            # 非单个大写字母的单词直接添加
            result.append(current_word)
            i += 1
    
    return ' '.join(result)

def clean_and_split_text(text):
    """
    1) 清洗只保留字母、数字、空格、常见标点符号(.,!?')，其余换空格
    2) 区分撇号/引号进行分句：引号内的 . ! 不触发分句，外部的 . ! 触发
    3) 去除最终句子里的“引号”字符，只保留撇号
    4) 新增特判：
       - 若句号前只有一个字母，就跳过这个句号，不作为分句边界
       - 若遇到 Mr. / st. / Dr. / Mrs. 等，也跳过句号
    """

    # 1. 简单清洗
    cleaned_text = re.sub(r"[^\w\s\.,!?:']", " ", text)
    
    sentences = []
    current_buffer = []

    i = 0
    while i < len(cleaned_text):
        ch = cleaned_text[i]

        
        # 但先对 '.' 做“是否跳过”的特判
        if ch == '.' :
            if should_skip_period(cleaned_text, i):
                # 跳过分句，直接把 '.' 当作普通字符处理
                current_buffer.append(ch)
                i += 1
                continue
            else:
                # 不跳过 => 把 '.' 加入当前句子并分句
                current_buffer.append(ch)
                sentence = "".join(current_buffer).strip()
                if sentence:
                    sentences.append(sentence)
                current_buffer = []
                i += 1
                continue

        elif ch == '!' :
            # 碰到 '!' 分句
            current_buffer.append(ch)
            sentence = "".join(current_buffer).strip()
            if sentence:
                sentences.append(sentence)
            current_buffer = []
            i += 1
            continue
        
        elif ch == '?' :
            # 碰到 '?' 分句
            current_buffer.append(ch)
            sentence = "".join(current_buffer).strip()
            if sentence:
                sentences.append(sentence)
            current_buffer = []
            i += 1
            continue
        
        else:
            # 其它情况正常加入字符
            current_buffer.append(ch)
            i += 1

    # 最后一段，若非空也作为一句
    leftover = "".join(current_buffer).strip()
    
    if leftover:
        sentences.append(leftover)

    
    
    # 3. 去掉句子中的“引号”，只保留撇号
    #    并且把 ! => .，把 : => , （如果你还需要这一步的话）
    sentences = [
        fix_spaced_caps(s.replace(':', ','))
        for s in sentences
    ]

    return sentences



def get_phone(text,language,version):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        
        phones, word2ph, norm_text = clean_text_inf_phone(formattext, language, version)
        
    return phones,None,norm_text

def get_bert_feature(text, word2ph, bert_model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)

def get_spepc(hps, filename):
    audio,_ = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):
        audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec

def load_audio(file, sr):
    try:
        file = clean_path(file)
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Error in audio loading")
    return np.frombuffer(out, np.float32).flatten()

def clean_path(path_str:str):
    if path_str.endswith(('\\','/')):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")

def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path, sr=None)
    return torch.FloatTensor(data), sampling_rate

def merge_sentences_minimum_n(sentences, n):
    """
    思路：
    1) 建一个缓冲区 buffer，不断往里添加句子
    2) 当 buffer 内的总词数 >= n 时，合并输出，然后清空 buffer
    3) 最后如果缓冲区还有剩余，则看你要不要跟上一条合并
    """
    merged_results = []
    buffer = []
    buffer_words = 0

    for s in sentences:
        # 先把当前句子放到缓冲区
        words_count = len(s.split())
        buffer.append(s)
        buffer_words += words_count

        # 如果达到/超过 n，则 flush
        if buffer_words >= n:
            merged_results.append(" ".join(buffer))
            buffer = []
            buffer_words = 0

    # 循环结束后，如果还有 leftover
    if buffer:
        # 你可以选择“单独输出”或“拼到上一条”
        leftover_words = sum(len(x.split()) for x in buffer)
        # 如果想“最后一条”也尽量 >= n，则可以跟上一条拼
        if leftover_words < n and merged_results:
            merged_results[-1] += " " + " ".join(buffer)
        else:
            merged_results.append(" ".join(buffer))

    return merged_results


def count_audio_tokens(strings):
    """
    使用 str.count('<|audio_token_') 来统计每个字符串中音频token的个数。
    """
    return [s.count('<|audio_token_') for s in strings]

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

if __name__ == '__main__':
    # s = '''Morning, Just listened to "AI Is Coming for Your Job—Elon Musk, OpenAI & The AI Arms Race" where Ben Horowitz casually dropped that AI models now use 600 BILLION variables! His laid-back explanation of how we went from 90% farming jobs to just 3% felt weirdly reassuring."'''
    s = '''Hi there! Today we dive into TIFF's $8B portfolio on "How I Invest with David Weisburd." Discover how nonprofits leverage private market strategies, independent sponsors, and anti-fragile portfolios for growth and resilience.'''
    s = s.replace("&", "and")
    print(s)
    s = s.replace("%", " percent")
    s = get_normed_text(s, 'en','v2')
    print(s)
    s = clean_and_split_text(s)
    print(s)
    # ['Hi there!', "Today we dive into TIFF's  8B portfolio on  How I Invest with David Weisburd."]
    s = merge_sentences_minimum_n(s,12)
    print(s)