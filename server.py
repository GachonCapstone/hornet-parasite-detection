import json
import base64
import threading
import time
import queue
import paho.mqtt.client as mqtt
import requests
from predict import predict_audio_bytes

# 요청 대기 큐
inference_queue = queue.Queue()

# 메시지 수신 시 실행될 콜백
def on_message(client, userdata, msg):
    print(f"[Receive] Topic: {msg.topic}")
    inference_queue.put((msg.topic, msg.payload))

# 추론 워커 (스레드)
def inference_worker():
    print("[Worker] Inference worker started.")
    while True:
        topic, payload = inference_queue.get()
        try:
            data = json.loads(payload.decode('utf-8'))
            audio_b64 = data.get('audio')
            if not audio_b64:
                print("[Worker] Error: 'audio' 키 없음")
                continue

            audio_bytes = base64.b64decode(audio_b64)
            results = predict_audio_bytes(
                audio_bytes=audio_bytes,
                model_path='bee_hornet_cnn.h5'
            )
            print(f"[Worker] Prediction done: {results}")

            # 웹 서버로 결과 전송
            response = requests.post(
                'http://localhost:8080/detect/hornet',
                json={'predictions': results},
                timeout=10.0
            )
            print(f"[Worker] POST responded {response.status_code}")
        except Exception as e:
            print(f"[Worker] Error: {e}")
        finally:
            inference_queue.task_done()

# 주기적 메시지 발행 스레드
def scheduled_publisher(client):
    topic = 'pi/request'
    while True:
        message = b'Scheduled request'
        client.publish(topic, message)
        print(f"[Publish] {topic}")
        time.sleep(60)

# 메인 함수
def main():
    client = mqtt.Client()
    client.on_message = on_message

    # 브로커 연결
    client.connect('localhost', 1883, 60)

    # 구독 설정
    client.subscribe('pi/response', qos=1)

    # 추론 워커 스레드 시작
    threading.Thread(target=inference_worker, daemon=True).start()

    # 발행 스레드 시작
    threading.Thread(target=scheduled_publisher, args=(client,), daemon=True).start()

    # 루프 시작
    client.loop_forever()

if __name__ == '__main__':
    main()
