import os
import json
import base64
import paho.mqtt.client as mqtt

BROKER_HOST = 'localhost'
BROKER_PORT = 1883
REQUEST_TOPIC = 'pi/request'
RESPONSE_TOPIC = 'pi/response'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, 'test_data')
IMAGE_EXTS = ['.jpg', '.jpeg', '.png']

def encode_file_to_b64(filepath: str) -> str:
    """파일을 base64로 인코딩하여 문자열로 반환."""
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

def find_paired_image(mp3_filename: str) -> str | None:
    """
    .mp3 파일명(예: 'abc.mp3')과 같은 이름의 이미지 파일을 TEST_DATA_DIR에서 찾음.
    """
    base = os.path.splitext(mp3_filename)[0]
    for ext in IMAGE_EXTS:
        candidate = os.path.join(TEST_DATA_DIR, base + ext)
        if os.path.isfile(candidate):
            return candidate
    return None

def send_all_pairs(client: mqtt.Client):
    """test_data 디렉토리 내 .mp3 + 매칭 이미지 쌍을 모두 RESPONSE_TOPIC으로 발행."""
    if not os.path.isdir(TEST_DATA_DIR):
        print(f"[Error] 디렉토리 없음: {TEST_DATA_DIR}")
        return

    for fname in os.listdir(TEST_DATA_DIR):
        if not fname.lower().endswith('.mp3'):
            continue

        audio_path = os.path.join(TEST_DATA_DIR, fname)
        image_path = find_paired_image(fname)
        if image_path is None:
            print(f"[Warning] 매칭 이미지 없음: {fname}")
            continue

        payload = {
            'audio': encode_file_to_b64(audio_path),
            'image': encode_file_to_b64(image_path),
        }
        raw = json.dumps(payload).encode('utf-8')
        client.publish(RESPONSE_TOPIC, raw, qos=1)
        print(f"[Publish] {RESPONSE_TOPIC} ← {fname} + {os.path.basename(image_path)}")

def on_request(client, userdata, msg):
    print(f"[Receive] {REQUEST_TOPIC} ← {msg.payload!r}")
    send_all_pairs(client)

def main():
    client = mqtt.Client()
    client.on_message = on_request

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.subscribe(REQUEST_TOPIC, qos=1)

    print(f"Listening for '{REQUEST_TOPIC}' …")
    client.loop_forever()

if __name__ == '__main__':
    main()
