import json
import base64
import threading
import time
import queue

import paho.mqtt.client as mqtt
import requests

from hornet_audio_detection import predict_audio_bytes
from hornet_image_detection import infer_image
from parasite_image_detection import parasite_detection
from datetime import datetime

# 요청 대기 큐
inference_queue = queue.Queue()

# 메시지 수신 시 실행될 콜백
def on_message(client, userdata, msg):
    print(f"[Receive] Topic: {msg.topic}")
    inference_queue.put((msg.topic, msg.payload))

def late_fusion(audio_results, image_results,
                w_audio: float = 0.4, w_image: float = 0.6) -> str:

    # 1) 최상위 하나만 꺼내오기
    a = audio_results[0]
    i = image_results[0]

    # 2) score 계산 (audio: a['prob'], image: i['confidence'])
    a_score = a.get('confidence', 0.0) * w_audio
    i_score = i.get('confidence', 0.0) * w_image

    # 3) 더 큰 쪽의 label 반환
    return a['predicted_label'] if a_score > i_score else i['predicted_label']


def inference_worker():
    print("[Worker] Inference worker started.")
    while True:
        topic, payload = inference_queue.get()
        try:
            data = json.loads(payload.decode('utf-8'))
            audio_b64 = data.get('audio')
            image_b64 = data.get('image')

            if not audio_b64 or not image_b64:
                print("[Worker] Error: 'audio' or 'image' key missing")
            else:
                # 1) 디코딩 & 추론
                audio_bytes = base64.b64decode(audio_b64)
                audio_results = predict_audio_bytes(audio_bytes=audio_bytes)
                print(f"[Worker] Audio probs: {audio_results}")

                image_bytes = base64.b64decode(image_b64)
                image_results = infer_image(
                    image_bytes=image_bytes,
                    conf_threshold=0.5
                )
                print(f"[Worker] Image probs: {image_results}")

                parasite_image_results = parasite_detection(image_bytes)
                print(f"[Worker] Parasite Image probs: {parasite_image_results}")

                # 2) late fusion으로 최종 클래스 결정
                final_label = late_fusion(audio_results, image_results)
                print(f"[Worker] Fused final label: {final_label}")

                # 3) 최종 레이블만 POST
                resp = requests.post(
                    'http://localhost:8080/detect/hornet',
                    json={'label': final_label, 'hornet_count': image_results[0].get('count'), 'parasite_count': parasite_image_results.get('count'), 'measuredAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')},
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
