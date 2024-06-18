from ultralytics import YOLO
import cv2
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
import tensorflow as tf

# TensorFlow 로깅 수준 설정 # 로깅이란 정보를 기록하는 것
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

model = YOLO("./yolov8n-face.pt")

# 사전에 훈련된 감정 분류 모델을 로드   
classifier = load_model(r'C:\Users\user\face_emotion_detection_CNN\model.h5')

# 감정 라벨 정의
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 감정 인식을 일정 간격으로 수행하기 위한 타이머 초기화
last_prediction_time = time.time()
prediction_interval = 0.5  # 500ms

emotion = None

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from webcam.")
        break
    
    
    results = model.predict(frame)

    # results가 리스트 형태로 반환되므로 각 요소에 대해 처리합니다.
    for result in results:
        boxes = result.boxes.xyxy

        # 감지된 바운딩 박스를 프레임에 그리면서 얼굴을 추출
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            frame = frame[y1:y2,x1:x2]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 얼굴 영역을 48x48 크기로 조정
            face = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

            if face.size != 0: # 얼굴 영역이 비어 있지 않을 때 실행
                
                # 얼굴 영역을 0~1 사이의 값으로 정규화합니다.
                roi = face.astype('float') / 255.0

                # 정규화된 얼굴 영역을 배열로 변환합니다.
                roi = img_to_array(roi)

                # 배열에 배치 차원을 추가합니다. 이 경우, 모델에 입력하기 위해 (1, height, width, channels) 형태로 변환됩니다.
                roi = np.expand_dims(roi, axis=0)

                # 모델을 사용하여 감정 예측을 수행합니다.
                prediction = classifier.predict(roi)[0]

                # 예측 결과에서 가장 높은 확률을 가진 감정 레이블을 선택합니다.
                label = emotion_labels[prediction.argmax()]
                label_position = (x1, y1 - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("YOLOv8 Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
