import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time
import tensorflow as tf
import mediapipe as mp

# TensorFlow 로깅 수준 설정 # 로깅이란 정보를 기록하는 것
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# 얼굴 인식 및 감정 인식 모델 초기화
cap = cv2.VideoCapture(0)  # 웹캠 사용
# mediapipe를 사용하여 얼굴을 인식하는 객체 생성
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 사전에 훈련된 감정 분류 모델을 로드   
classifier = load_model(r'C:\Users\user\face_emotion_detection_CNN\model.h5')

# 감정 라벨 정의
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 감정 인식을 일정 간격으로 수행하기 위한 타이머 초기화
last_prediction_time = time.time()
prediction_interval = 0.5  # 500ms

emotion = None  # 감정 변수 초기화

while True:
    
    # 웹캠에서 한 프레임씩 읽어오기
    _, frame = cap.read()
    
    # 감정을 인식할 얼굴을 흑백 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face_landmarks = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 첫 번째 얼굴만 사용하여 사각형 그리기
    if results.detections and face_landmarks.multi_face_landmarks:
        try:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame.shape[1])
                y = int(bbox.ymin * frame.shape[0])
                w = int(bbox.width * frame.shape[1])
                h = int(bbox.height * frame.shape[0])
                # 검출된 얼굴에 사각형 그리기
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                # 얼굴 영역을 흑백 이미지로 자르기
                roi_gray = gray[y:y+h, x:x+w]
                # 얼굴 영역을 48x48 크기로 조정
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                # 인식된 얼굴 영역이 있는 경우 감정 예측 수행
                if np.sum([roi_gray]) != 0:
                    # 감정 예측을 일정 간격으로 수행
                    current_time = time.time()
                    if current_time - last_prediction_time > prediction_interval:
                        # 이미지 정규화
                        roi = roi_gray.astype('float') / 255.0
                        # 이미지를 배열로 변환
                        roi = img_to_array(roi)
                        # 모델 입력 형식에 맞게 차원 확장
                        roi = np.expand_dims(roi, axis=0)

                        # 모델로 감정 예측
                        prediction = classifier.predict(roi)[0]
                        # 가장 높은 확률의 감정 라벨 선택
                        emotion = emotion_labels[prediction.argmax()]
                        # 감정 라벨 위치 설정
                        label_position = (x, y)
                        # 감정 라벨 텍스트를 프레임에 추가
                        cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # 예측 시간을 업데이트
                        last_prediction_time = current_time
                        # 감정을 출력
                        print(f"Detected Emotion: {emotion}")
                    else:
                        # 이전 예측을 그대로 사용
                        if emotion:
                            cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # 얼굴 영역이 없는 경우 텍스트 추가
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            # 얼굴이 화면 밖으로 나가는 경우 예외 처리
            pass
    else:
        # 얼굴이 검출되지 않은 경우 텍스트 추가
        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면에 감정 인식 결과를 보여주기
    cv2.imshow('Emotion Detector', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
