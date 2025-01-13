import requests
import json
import tempfile
import os
from scipy.io import wavfile
from scipy import signal
import numpy as np

engine_api = "http://127.0.0.1:50032"

# 音声を合成する関数
def tts(text, speaker, engine):
    if engine == "voicevox":
        return voicevox(text, speaker)
    if engine == "coeiroink":
        speaker = "3c37646f-3881-5374-2a83-149267990abc"
        styleId = 0
        return coeiroink(styleId, speaker, text)

def voicevox(text, speaker):
    # 音声合成用のクエリ作成
    query = requests.post(
        f'http://127.0.0.1:50021/audio_query',
        params=(('text', text),('speaker', speaker),)
    )
    
    # WAVで音声合成
    synthesis = requests.post(
        f'http://127.0.0.1:50021/synthesis',
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},  # WAVフォーマットを指定
        params=(('text', text),('speaker', speaker),),
        data=json.dumps(query.json())
    )
    
    # 一時ファイルにWAVを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(synthesis.content)
        temp_file_path = temp_file.name

    # WAVファイルを読み込んでPCMデータに変換
    try:
        sample_rate, data = wavfile.read(temp_file_path)
        # 必要に応じてサンプルレートを変換（24000Hzに）
        if sample_rate != 24000:
            # ここでリサンプリングが必要な場合は実装する
            pass
        # int16形式に変換
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
        os.unlink(temp_file_path)  # 一時ファイルを削除
        return data.tobytes()
    except Exception as e:
        os.unlink(temp_file_path)  # エラー時も一時ファイルを削除
        raise e

def resample(data, src_rate, dst_rate):
    """音声データをリサンプリングする"""
    number_of_samples = round(len(data) * float(dst_rate) / src_rate)
    return signal.resample(data, number_of_samples)

def coeiroink(styleId, speaker, text):
    # 音声合成（WAV形式で取得）
    response = requests.post(
        engine_api + '/v1/predict',
        json={
            'text': text,
            'speakerUuid': speaker,
            'styleId': styleId,
            'prosodyDetail': None,
            'speedScale': 1,
        }
    )
    
    # 一時ファイルにWAVを保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    try:
        sample_rate, data = wavfile.read(temp_file_path)
        
        # リサンプリング処理
        if sample_rate != 24000:
            data = resample(data, sample_rate, 24000)
        
        # int16形式に変換
        if data.dtype == np.float32:
            data = (data * 32767.0).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)
        
        os.unlink(temp_file_path)
        return data.tobytes()
    except Exception as e:
        os.unlink(temp_file_path)
        raise e
