import os
import requests
import subprocess
import time
from influxdb_client import InfluxDBClient

# 環境変数の読み込み
MODEL = os.getenv("OLLAMA_MODEL")
STT_URL = os.getenv("STT_URL")
TTS_URL = os.getenv("TTS_URL")
SPEAKER_ID = os.getenv("SPEAKER_ID")

# スクリーンセーバー制御
def control_screen(action):
    if action == "wake":
        subprocess.run(["xscreensaver-command", "-deactivate"], env={"DISPLAY": ":0"})
    else:
        subprocess.run(["xscreensaver-command", "-activate"], env={"DISPLAY": ":0"})

# VOICEVOXで喋る
def speak(text):
    query = requests.post(f"{TTS_URL}/audio_query?text={text}&speaker={SPEAKER_ID}")
    audio = requests.post(f"{TTS_URL}/synthesis?speaker={SPEAKER_ID}", data=query.content)
    with open("reply.wav", "wb") as f:
        f.write(audio.content)
    subprocess.run(["aplay", "reply.wav"])

# メイン処理
def jarvis_cycle():
    # 本来はここに openWakeWord の検知を入れる
    print("Waiting for 'Jarvis'...")
    
    # 擬似的に検知した後の流れ
    control_screen("wake")
    speak("はい、Sir。何を確認しますか？")
    
    # 録音 -> STT -> Ollama -> InfluxDB -> TTS の流れをここに実装
    # Llama 3.2 3B は Fluxクエリの生成も高速です

if __name__ == "__main__":
    jarvis_cycle()
