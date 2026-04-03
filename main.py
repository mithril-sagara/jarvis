import os
import requests
import subprocess
import time
from influxdb_client import InfluxDBClient

# --- 1. 環境変数の読み込み ---
URL = os.getenv("INFLUXDB_URL")
TOKEN = os.getenv("INFLUXDB_TOKEN")
ORG = os.getenv("INFLUXDB_ORG")
BUCKET = os.getenv("INFLUXDB_BUCKET", "hems")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TTS_URL = os.getenv("TTS_URL", "http://tts:50021")
SPEAKER_ID = os.getenv("SPEAKER_ID", "13") # 青山龍星

# InfluxDBクライアントの初期化
client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)

# --- 2. 画面制御 ---
def control_screen(action):
    try:
        if action == "wake":
            subprocess.run(["xscreensaver-command", "-deactivate"], env={"DISPLAY": ":0"}, check=False)
            subprocess.run(["xset", "dpms", "force", "on"], env={"DISPLAY": ":0"}, check=False)
        else:
            subprocess.run(["xscreensaver-command", "-activate"], env={"DISPLAY": ":0"}, check=False)
    except Exception as e:
        print(f"Screen control error: {e}")

# --- 3. InfluxDBから最新の発電量を取得 ---
def get_current_solar():
    try:
        query = f'from(bucket: "{BUCKET}") |> range(start: -1m) |> filter(fn: (r) => r["_field"] == "solar_w") |> last()'
        tables = client.query_api().query(query, org=ORG)
        for table in tables:
            for record in table.records:
                return record.get_value()
        return 0
    except Exception as e:
        print(f"InfluxDB error: {e}")
        return None

# --- 4. Llama 3.2 に応答文を作らせる ---
def ask_jarvis(user_text):
    solar_val = get_current_solar()
    solar_str = f"{solar_val}W" if solar_val is not None else "不明"
    
    # ジャービスのキャラ設定プロンプト
    system_prompt = (
        "あなたはアイアンマンのトニー・スタークに仕えるAI、ジャービスです。 "
        "冷静沈着で、丁寧な執事口調で話してください。 "
        f"現在の太陽光発電量は {solar_str} です。この数値を元に回答してください。 "
        "回答は30文字以内で、簡潔に数値を含めてください。"
    )

    payload = {
        "model": MODEL,
        "prompt": f"{system_prompt}\nユーザー: {user_text}\nジャービス:",
        "stream": False
    }
    
    try:
        response = requests.post(f"http://{OLLAMA_HOST}/api/generate", json=payload)
        return response.json().get("response", "エラーが発生しました、Sir。")
    except Exception as e:
        return f"AI通信エラーです: {e}"

# --- 5. VOICEVOX で発話 ---
def speak(text):
    try:
        # 音声合成クエリ作成
        query_res = requests.post(f"{TTS_URL}/audio_query?text={text}&speaker={SPEAKER_ID}")
        # 音声データ生成
        audio_res = requests.post(f"{TTS_URL}/synthesis?speaker={SPEAKER_ID}", data=query_res.content)
        
        with open("reply.wav", "wb") as f:
            f.write(audio_res.content)
        
        # aplay で再生
        subprocess.run(["aplay", "reply.wav"], check=False)
    except Exception as e:
        print(f"TTS error: {e}")

# --- 6. メインサイクル ---
def jarvis_cycle():
    print("Jarvis System Online. Ready for command.")
    
    while True:
        # シミュレーション：Enterキーで「ジャービス！」と呼びかけたとみなす
        input("\n[Enter]キーを押して呼びかけ（ジャービス！）をシミュレート...")
        
        # 1. 画面を復帰
        control_screen("wake")
        
        # 2. 思考（InfluxDB値取得含む）
        print("Thinking...")
        answer = ask_jarvis("現在の発電状況を教えてください。")
        
        # 3. 発話
        print(f"Jarvis: {answer}")
        speak(answer)
        
        # 4. 少し待ってから終了（実際はここからまたループに戻る）
        print("Waiting for next command...")

if __name__ == "__main__":
    jarvis_cycle()
