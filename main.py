import time
import subprocess
from ollama import Client

# 設定
WAKE_WORD = "jarvis"
client = Client(host='http://jarvis-brain:11434')

def start_screensaver():
    # スクリーンセーバーを起動するコマンド（ホスト側と通信が必要）
    subprocess.run(["xscreensaver-command", "-activate"])

def stop_screensaver():
    # スクリーンセーバーを停止（復帰）
    subprocess.run(["xscreensaver-command", "-deactivate"])

def main_loop():
    while True:
        print("待機中: ジャービスを待っています...")
        # 1. WakeWord検知 (openWakeWordのストリームを監視)
        if wait_for_wake_word(WAKE_WORD):
            stop_screensaver() # 画面復帰
            play_sound("detect.wav") # 起動音「ピッ」
            
            # 2. 音声録音 & Faster-Whisperでテキスト化
            user_text = listen_and_stt()
            
            # 3. Llama 3.1 で思考（必要ならInfluxDBをクエリ）
            # ここで Function Calling を実装
            response_text = think_with_llama(user_text)
            
            # 4. Piper で発話
            speak_with_piper(response_text)
            
            # 5. 完了後、少し待ってセーバーに戻る
            time.sleep(10)
            start_screensaver()

if __name__ == "__main__":
    main_loop()
