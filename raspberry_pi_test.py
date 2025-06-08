# 실제 라즈베리파이가 아닌, pc에서 mp4 파일을 스트리밍이라고 가정하고 실행하는 코드
import os
import io
import time
import json
import base64
import glob
import threading
import tempfile
import subprocess

import cv2
import paho.mqtt.client as mqtt

# === 설정 ===
HIVE_ID        = 1
BROKER_HOST    = 'localhost'
BROKER_PORT    = 1883
REQUEST_TOPIC  = 'pi/request'
RESPONSE_TOPIC = 'pi/response'
VIDEO_DIR      = 'test_video'

# === 전역 상태 변수 및 락 ===
state_lock = threading.Lock()
current_frame = None        # numpy.ndarray
current_video = None        # str, 파일 경로
current_time = 0.0          # float, 초 단위 타임스탬프
current_fps = 30.0          # 재생 FPS (영상마다 다를 수 있음)

def encode_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')

def extract_audio_segment(video_path: str, start: float, duration: float = 3.0) -> bytes:
    """FFmpeg로 video_path에서 start부터 duration 만큼 오디오를 잘라 MP3로 리턴."""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tf:
        tmp_mp3 = tf.name
    cmd = [
        'ffmpeg',
        '-ss', str(start),
        '-t', str(duration),
        '-i', video_path,
        '-vn',                # 비디오 스트림 제외
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',
        '-y', tmp_mp3
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    with open(tmp_mp3, 'rb') as f:
        data = f.read()
    os.remove(tmp_mp3)
    return data

def on_request(client, userdata, msg):
    print(f"[Request Received] topic={msg.topic}")
    # 현재 재생 상태 캡처
    with state_lock:
        frame = current_frame.copy() if current_frame is not None else None
        video = current_video
        timestamp = current_time

    if frame is None or video is None:
        print("⚠️ 아직 영상이 준비되지 않았습니다.")
        return

    # 1) 오디오 추출
    audio_bytes = extract_audio_segment(video, timestamp, duration=3.0)

    # 2) 프레임을 JPEG로 인코딩
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        print("⚠️ 프레임 인코딩 실패")
        return
    img_bytes = buf.tobytes()

    # 3) 페이로드 생성 및 발행
    payload = {
        'id': HIVE_ID,
        'image': encode_to_b64(img_bytes),
        'audio': encode_to_b64(audio_bytes),
    }
    raw = json.dumps(payload).encode('utf-8')
    client.publish(RESPONSE_TOPIC, raw, qos=1)
    print(f"[Published] {RESPONSE_TOPIC}")

def video_playback_loop():
    global current_frame, current_video, current_time, current_fps

    # MQTT 클라이언트 백그라운드 루프 시작
    client = mqtt.Client()
    client.on_message = on_request
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.subscribe(REQUEST_TOPIC, qos=1)
    client.loop_start()

    # 무한 루프: 모든 비디오를 다 본 뒤 다시 처음으로
    while True:
        for video_path in sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⛔️ 영상 열기 실패: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            current_fps = fps
            frame_time = 1.0 / fps
            print(f"▶ 재생 시작: {os.path.basename(video_path)} ({fps:.1f} FPS)")

            start_ts = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 전역 상태 업데이트
                elapsed = time.time() - start_ts
                with state_lock:
                    current_frame = frame
                    current_video = video_path
                    current_time = elapsed

                # 화면에 출력
                cv2.imshow('Video Playback', frame)
                if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    client.loop_stop()
                    return

            cap.release()

if __name__ == '__main__':
    video_playback_loop()