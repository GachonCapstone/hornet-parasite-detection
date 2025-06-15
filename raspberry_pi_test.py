import os
import io
import time
import json
import base64
import glob
import threading
import tempfile
import subprocess
import signal
import sys

import cv2
import paho.mqtt.client as mqtt
from flask import Flask, Response

# === 설정 ===
HIVE_ID        = 1
BROKER_HOST    = 'localhost'
BROKER_PORT    = 1883
REQUEST_TOPIC  = 'pi/request'
RESPONSE_TOPIC = 'pi/response'
VIDEO_DIR      = 'test_video'

# === 전역 상태 변수 및 락 ===
state_lock = threading.Lock()
current_frame = None
current_video = None
current_time = 0.0
current_fps = 30.0

# 종료 플래그
stop_event = threading.Event()

# MQTT 클라이언트 전역 참조
mqtt_client = None

# ===== MQTT 요청 → 이미지/오디오 페이로드 발행 =====
def encode_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')

def extract_audio_segment(video_path: str, start: float, duration: float = 3.0) -> bytes:
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tf:
        tmp_mp3 = tf.name
    cmd = ['ffmpeg', '-ss', str(start), '-t', str(duration),
           '-i', video_path, '-vn', '-codec:a', 'libmp3lame',
           '-qscale:a', '2', '-y', tmp_mp3]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    with open(tmp_mp3, 'rb') as f:
        data = f.read()
    os.remove(tmp_mp3)
    return data

def on_request(client, userdata, msg):
    with state_lock:
        frame = current_frame.copy() if current_frame is not None else None
        video = current_video
        timestamp = current_time

    if frame is None or video is None:
        return

    audio_bytes = extract_audio_segment(video, timestamp)
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return
    img_bytes = buf.tobytes()

    payload = {
        'id': HIVE_ID,
        'image': encode_to_b64(img_bytes),
        'audio': encode_to_b64(audio_bytes),
    }
    mqtt_client.publish(RESPONSE_TOPIC, json.dumps(payload).encode('utf-8'), qos=1)

# ===== 비디오 재생 루프 =====
def video_playback_loop():
    global current_frame, current_video, current_time, current_fps, mqtt_client

    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_request
    mqtt_client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    mqtt_client.subscribe(REQUEST_TOPIC, qos=1)
    mqtt_client.loop_start()

    try:
        while not stop_event.is_set():
            for video_path in sorted(glob.glob(os.path.join(VIDEO_DIR, 'bee_parasite_sample_video.mp4'))):
                if stop_event.is_set():
                    break

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                current_fps = fps
                frame_time = 1.0 / fps

                start_ts = time.time()
                while cap.isOpened() and not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    elapsed = time.time() - start_ts
                    with state_lock:
                        current_frame = frame
                        current_video = video_path
                        current_time = elapsed

                    cv2.imshow('Video Playback', frame)
                    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
                        stop_event.set()
                        break

                cap.release()
    finally:
        # 정리
        mqtt_client.loop_stop()
        cv2.destroyAllWindows()

# ===== Flask 앱 & 스트리밍 =====
app = Flask(__name__)

def generate_mjpeg():
    while not stop_event.is_set():
        with state_lock:
            frame = current_frame.copy() if current_frame is not None else None
            fps = current_fps
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                jpg = buf.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(1.0 / max(fps, 1.0))

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== SIGINT 핸들러 =====
def handle_sigint(signum, frame):
    print("\nShutting down...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

# ===== 진입점 =====
if __name__ == '__main__':
    # 1) 비디오 재생/ MQTT 스레드
    threading.Thread(target=video_playback_loop, daemon=True).start()

    # 2) Flask 실행
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        # Flask가 멈추면 stop_event도 설정
        stop_event.set()
        print("Exited cleanly.")
        sys.exit(0)
