import os
import requests
import subprocess
import time
import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from influxdb_client import InfluxDBClient

# --- 1. 環境変数の読み込み ---
URL = os.getenv("INFLUXDB_URL")
TOKEN = os.getenv("INFLUXDB_TOKEN")
ORG = os.getenv("INFLUXDB_ORG")
BUCKET = os.getenv("INFLUXDB_BUCKET", "hems")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
STT_URL = os.getenv("STT_URL")
TTS_URL = os.getenv("TTS_URL")
SPEAKER_ID = os.getenv("SPEAKER_ID", "13") 
MODEL_PATH = "/app/models/jarvis.tflite"

# --- 2. 各種設定と初期化 ---
CHANNELS = 1
RATE = 16000
CHUNK = 1280 

# InfluxDBクライアント初期化
client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)

print(f"--- Jarvis System Booting ---")

# モデルファイルの存在チェック（ここが重要）
if not os.path.exists(MODEL_PATH):
    print(f"❌ FATAL ERROR: Model file not found at {MODEL_PATH}")
    # ファイルがない場合はライブラリ内の標準モデルを探すようにフォールバック（保険）
    oww_model = Model(wakeword_models=["alexa"], inference_framework="tflite")
    print("⚠️ Warning: Falling back to 'alexa' model.")
else:
    print(f"✅ Found model: {MODEL_PATH}")
    # 修正：絶対パスをリストで渡し、推論エンジンを明示
    oww_model = Model(
        wakeword_models=[MODEL_PATH], 
        inference_framework="tflite"
    )

# マイクストリーム開始
try:
    audio_py = pyaudio.PyAudio()
    mic_stream = audio_py.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                               input=True, frames_per_buffer=CHUNK)
    print("✅ Microphone stream started.")
except Exception as e:
    print(f"❌ Microphone Error: {e}")

# --- 3. 画面制御 ---
def control_screen(action):
    try:
        env = {"DISPLAY": ":0"}
        if action == "wake":
            subprocess.run(["xscreensaver-command", "-deactivate"], env=env, check=False)
            subprocess.run(["xset", "dpms", "force", "on"], env=env, check=False)
        else:
            subprocess.run(["xscreensaver-command", "-activate"], env=env, check=False)
    except Exception as e:
        print(f"Screen Control Error: {e}")

# --- 4. 発電量取得 ---
def get_current_solar():
    try:
        query = f'from(bucket: "{BUCKET}") |> range(start: -1m) |> filter(fn: (r) => r["_field"] == "solar_w") |> last()'
        tables = client.query_api().query(query, org=ORG)
        for table in tables:
            for record in table.records:
                return record.get_value()
        return 0
    except Exception as e:
        print(f"InfluxDB Query Error: {e}")
        return 0

# --- 5. 耳 (STT) ---
def listen_and_stt():
    print("🎤 Listening...")
    frames = []
    # 3秒間サンプリング
    for _ in range(0, int(RATE / CHUNK * 3)):
        data = mic_stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    audio_data = b''.join(frames)
    try:
        files = {'audio_file': ('speech.wav', audio_data, 'audio/wav')}
        res = requests.post(STT_URL, files=files, timeout=10)
        text = res.json().get("text", "")
        return text
    except Exception as e:
        print(f"STT Error: {e}")
        return ""

# --- 6. 脳 (LLM) ---
def ask_jarvis(user_text):
    solar_val = get_current_solar()
    system_prompt = (
        "あなたはジャービスです。冷静で丁寧な執事口調で。 "
        f"現在の太陽光発電量は {solar_val}W です。回答は30文字以内で簡潔に数値を含めてください。"
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{system_prompt}\nユーザー: {user_text}\nジャービス:",
        "stream": False
    }
    try:
        res = requests.post(f"http://{OLLAMA_HOST}/api/generate", json=payload, timeout=15)
        return res.json().get("response", "Sir, 申し訳ありません。エラーです。")
    except Exception as e:
        print(f"LLM Error: {e}")
        return "通信に失敗しました。"

# --- 7. 口 (TTS) ---
def speak(text):
    try:
        # VOICEVOXへのクエリ
        q_res = requests.post(f"{TTS_URL}/audio_query?text={text}&speaker={SPEAKER_ID}", timeout=10)
        # 音声合成
        a_res = requests.post(f"{TTS_URL}/synthesis?speaker={SPEAKER_ID}", data=q_res.content, timeout=20)
        with open("reply.wav", "wb") as f:
            f.write(a_res.content)
        # 再生
        subprocess.run(["aplay", "reply.wav"], check=False)
    except Exception as e:
        print(f"TTS/Audio Error: {e}")

# --- 8. メインループ ---
def jarvis_cycle():
    print("\n🟢 Jarvis Online. Waiting for 'Hey Jarvis'...") 
    
    while True:
        try:
            data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_frame = np.frombuffer(data, dtype=np.int16)
            
            # ウェイクワード推論
            prediction = oww_model.predict(audio_frame)
            
            # 全モデルの判定をチェック
            detected = False
            for mdl in prediction:
                if prediction[mdl] > 0.5:
                    detected = True
                    print(f"✨ Detected {mdl} ({prediction[mdl]:.2f})")
                    break

            if detected:
                control_screen("wake")
                
                user_command = listen_and_stt()
                if user_command.strip():
                    print(f"👤 You: {user_command}")
                    answer = ask_jarvis(user_command)
                    print(f"🤖 Jarvis: {answer}")
                    speak(answer)
                else:
                    print("... (No command detected)")
                
                time.sleep(1)
                print("\n🟢 Waiting for 'Hey Jarvis'...")

        except Exception as e:
            print(f"Main Loop Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    jarvis_cycle()
