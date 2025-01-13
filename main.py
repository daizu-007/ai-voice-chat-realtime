# PythonからGemini 2.0 flash expのリアルタイム会話を行うコード
# https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_starter.py
# 上記のコードを参考にしています

# "GEMINI-API-TOKEN" 環境変数にGemini APIのトークンを設定してください

# ライブラリのインポート
import asyncio # 非同期処理のために使用
import traceback
import pyaudio
from dotenv import load_dotenv
import os
from synthesis import tts  # 追加

from google import genai # Googleの最新のAIライブラリ。今回のメイン

load_dotenv() # 環境変数を読み込む

# 音声関連の設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
# モデルの設定
MODEL = "models/gemini-2.0-flash-exp" # 現在リアルタイム会話に対応しているモデルはこれだけです
GEMINI_API_KEY = os.getenv("GEMINI-API-TOKEN")

client = genai.Client(api_key=GEMINI_API_KEY ,http_options={"api_version": "v1alpha"}) # リアルタイム会話を行うためにはv1alphaを指定する必要があります
CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}
py_audio = pyaudio.PyAudio()

class Main:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.text_buffer = ""  # テキストバッファを追加

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = py_audio.get_default_input_device_info()
        # ブロッキングI/Oを他のスレッドで行う
        audio_stream = await asyncio.to_thread(
            py_audio.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            # 発話開始と終了の検出はしてないっぽい？
            # コードを見る限りGeminiってリアルタイムで送られてくる音声ストリームを処理してるのかも
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            self.text_buffer = ""  # ターン開始時にバッファをクリア
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    # Geminiネイティブの音声出力の場合
                    self.audio_in_queue.put_nowait(data)
                elif text := response.text:
                    print(text, end="")
                    self.text_buffer += text  # テキストを蓄積

            # ターン終了時に音声合成
            if self.voice_mode == "2" and self.text_buffer.strip():
                try:
                    audio_data = await asyncio.to_thread(
                        tts,
                        text=self.text_buffer,
                        speaker=1,
                        engine="coeiroink"
                    )
                    self.audio_in_queue.put_nowait(audio_data)
                except Exception as e:
                    print(f"\n音声合成エラー: {e}")

            # キューを空にする
            # これはplay_audioの後に処理される？
            await asyncio.sleep(0.1)
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            py_audio.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        self.voice_mode = None
        while self.voice_mode not in ["1", "2"]:
            if self.voice_mode:
                print("無効な入力です。1、もしくは2を入力してください")
            print("使用する音声合成モデルを選択してください(1: Geminiネイティブ出力, 2: COEIROINK)")
            self.voice_mode = input("> ")
        if self.voice_mode == "1":
            CONFIG["generation_config"]["response_modalities"] = ["AUDIO"]
        else:
            CONFIG["generation_config"]["response_modalities"] = ["TEXT"]
        try:
            # TaskGroupで並行処理をさせるらしい
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)

def main():
    loop = Main()
    asyncio.run(loop.run())

if __name__ == "__main__":
    main()