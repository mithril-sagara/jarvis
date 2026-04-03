import os
import requests
import subprocess
import time
import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from influxdb_client import InfluxDBClient

# --- 1. 環境変数の読み込み (変更なし) ---
URL = os.getenv("INFLUXDB_URL")
TOKEN = os.getenv("INFLUXDB_TOKEN")
ORG = os.getenv("INFLUXDB_ORG")
BUCKET = os.getenv("INFLUXDB_BUCKET", "hems")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b") # 爆速版に変更
STT_URL = os.getenv("STT_URL")
TTS_URL = os.getenv("TTS_URL")
SPEAKER_ID = os.getenv("SPEAKER_ID", "13") 
MODEL_PATH = "/app/models/jarvis.tflite" # 自動取得したモデルを使用

# --- 2. 各種設定と初期化 (ジャービスに修正) ---
CHANNELS = 1
RATE = 16000
CHUNK = 1280 

client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)

# モデルを「アレクサ」から「ジャービス」に変更
oww_model = Model(wakeword_models=[MODEL_PATH], inference_framework="tflite")

audio_py = pyaudio.PyAudio()
mic_stream = audio_py.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                           input=True, frames_per_buffer=CHUNK)

# --- 3. 画面制御 (変更なし) ---
def control_screen(action):
    try:
        if action == "wake":
            subprocess.run(["xscreensaver-command", "-deactivate"], env={"DISPLAY": ":0"}, check=False)
            subprocess.run(["xset", "dpms", "force", "on"], env={"DISPLAY": ":0"}, check=False)
        else:
            subprocess.run(["xscreensaver-command", "-activate"], env={"DISPLAY": ":0"}, check=False)
    except: pass

# --- 4. InfluxDBから最新の発電量を取得 (変更なし) ---
def get_current_solar():
    try:
        query = f'from(bucket: "{BUCKET}") |> range(start: -1m) |> filter(fn: (r) => r["_field"] == "solar_w") |> last()'
        tables = client.query_api().query(query, org=ORG)
        for table in tables:
            for record in table.records:
                return record.get_value()
        return 0
    except: return 0

# --- 5. 耳 (STT): 呼びかけ後の音声をテキスト化 (変更なし) ---
def listen_and_stt():
    print("Listening to your command...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * 3)):
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    audio_data = b''.join(frames)
    try:
        files = {'audio_file': ('speech.wav', audio_data, 'audio/wav')}
        res = requests.post(STT_URL, files=files)
        return res.json().get("text", "")
    except: return ""

# --- 6. 脳 (LLM): Llama 3.2 への問い合わせ (変更なし) ---
def ask_jarvis(user_text):
    solar_val = get_current_solar()
    system_prompt = (
        "あなたはジャービスです。冷静で丁寧な執事口調で話してください。 "
        f"現在の太陽光発電量は {solar_val}W です。回答は30文字以内で簡潔に数値を含めてください。"
    )
    payload = {
        "model": MODEL,
        "prompt": f"{system_prompt}\nユーザー: {user_text}\nジャービス:",
        "stream": False
    }
    try:
        res = requests.post(f"http://{OLLAMA_HOST}/api/generate", json=payload)
        return res.json().get("response", "Sir, 申し訳ありません。エラーです。")
    except: return "通信に失敗しました。"

# --- 7. 口 (TTS): VOICEVOX で発話 (変更なし) ---
def speak(text):
    try:
        q_res = requests.post(f"{TTS_URL}/audio_query?text={text}&speaker={SPEAKER_ID}")
        a_res = requests.post(f"{TTS_URL}/synthesis?speaker={SPEAKER_ID}", data=q_res.content)
        with open("reply.wav", "wb") as f:
            f.write(a_res.content)
        subprocess.run(["aplay", "reply.wav"], check=False)
    except: print("TTS Error")

# --- 8. メインループ (判定部分のみ強化) ---
def jarvis_cycle():
    # 表示を Jarvis に合わせて変更
    print("Jarvis Online. Waiting for 'Hey Jarvis'...") 
    
    while True:
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        audio_frame = np.frombuffer(data, dtype=np.int16)
        
        # ウェイクワードの推論
        prediction = oww_model.predict(audio_frame)
        
        # ★ 修正箇所：
        # 公式モデルは内部で 'hey_jarvis' という名前で保持されています。
        # 単純に「0.5を超えたモデルが一つでもあるか」をチェックする書き方にします。
        detected = False
        for mdl in prediction:
            if prediction[mdl] > 0.5: # 感度設定（0.5）
                detected = True
                break

        if detected:
            print("Wake Word Detected!")
            control_screen("wake")
            
            # 1. あなたの声を聴く (STT)
            user_command = listen_and_stt()
            print(f"You said: {user_command}")
            
            if user_command:
                # 2. AIが考える (LLM)
                answer = ask_jarvis(user_command)
                print(f"Jarvis: {answer}")
                
                # 3. AIが喋る (TTS)
                speak(answer)
            
            time.sleep(2)
            print("Waiting for Wake Word...")
          
if __name__ == "__main__":
    jarvis_cycle()
