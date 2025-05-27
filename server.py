import json
import base64
import threading
import time
import queue

import paho.mqtt.client as mqtt
import requests

from audio_detection import predict_audio_bytes
from image_detection import infer_image 

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
            image_b64 = data.get('image')

            if not audio_b64 or not image_b64:
                print("[Worker] Error: 'audio' 키 없음")
            else:
                audio_bytes = base64.b64decode(audio_b64)
                audio_results = predict_audio_bytes(
                    audio_bytes=audio_bytes,
                    model_path='audio_detection_model.h5'
                )
                print(f"[Worker] Audio prediction done: {audio_results}")

                image_bytes = base64.b64decode(image_b64)
                image_results = infer_image(
                    image_bytes=image_bytes,
                    conf_threshold=0.5
                )
                print(f"[Worker] Image prediction done: {image_results}")

                # 결과 POST
                resp = requests.post(
                    'http://localhost:8080/detect/hornet',
                    json={'audio_predictions': audio_results, 'image_predictions':image_results},
                    timeout=10.0
                )
                print(f"[Worker] POST responded {resp.status_code}")
            
            

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
