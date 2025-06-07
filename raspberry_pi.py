import io
import time
import os
import json
import base64
import subprocess

from picamera import PiCamera
import paho.mqtt.client as mqtt

# === 설정 ===
HIVE_ID        = 1
BROKER_HOST    = 'localhost'    # 브로커 주소
BROKER_PORT    = 1883
REQUEST_TOPIC  = 'pi/request'
RESPONSE_TOPIC = 'pi/response'

# === 미디어 캡처 함수 ===
def capture_image() -> bytes:
    """PiCamera로 사진 한 장을 JPEG 포맷으로 캡처하여 바이트로 반환."""
    stream = io.BytesIO()
    with PiCamera() as camera:
        camera.resolution = (1024, 768)
        time.sleep(2)  # 카메라 워밍업
        camera.capture(stream, format='jpeg')
    return stream.getvalue()

def record_audio(duration: int = 3,
                 wav_path: str = '/tmp/audio.wav',
                 mp3_path: str = '/tmp/audio.mp3') -> bytes:
    """
    arecord로 WAV로 녹음한 뒤 ffmpeg로 MP3로 인코딩
    """
    # 1) WAV로 녹음
    subprocess.run([
        'arecord',
        '-D', 'plughw:1,0',  # 마이크 디바이스 이름은 상황에 맞게 조정
        '-f', 'cd',
        '-t', 'wav',
        '-d', str(duration),
        wav_path
    ], check=True)
    # 2) MP3로 변환 (ffmpeg 필요)
    subprocess.run([
        'ffmpeg', '-y',
        '-i', wav_path,
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',
        mp3_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    with open(mp3_path, 'rb') as f:
        return f.read()

def encode_to_b64(data: bytes) -> str:
    """바이트 데이터를 Base64 문자열로 인코딩"""
    return base64.b64encode(data).decode('ascii')

# === MQTT 콜백 ===
def on_request(client, userdata, msg):
    print(f"[Request Received] {msg.topic}")
    # 1) 미디어 캡처
    img = capture_image()
    audio = record_audio()

    # 2) 페이로드 구성
    payload = {
        'id': HIVE_ID,
        'image': encode_to_b64(img),
        'audio': encode_to_b64(audio),
    }
    raw = json.dumps(payload).encode('utf-8')

    # 3) 응답 발행
    client.publish(RESPONSE_TOPIC, raw, qos=1)
    print(f"[Published] {RESPONSE_TOPIC}")

# === 메인 ===
def main():
    client = mqtt.Client()
    client.on_message = on_request

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.subscribe(REQUEST_TOPIC, qos=1)

    print(f"Listening for requests on '{REQUEST_TOPIC}' …")
    client.loop_forever()

if __name__ == '__main__':
    main()