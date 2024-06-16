from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Haar cascade classifier를 사용하여 얼굴을 인식하는 객체 생성
face_classifier = cv2.CascadeClassifier(r'C:\Users\jsy99\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')

# 사전에 훈련된 감정 분류 모델을 로드
classifier = load_model(r'C:\Users\jsy99\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

# 감정 라벨 정의
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) 


while True:
    
    # 웹캠에서 한 프레임씩 읽어오기
    _, frame = cap.read()
    
    # 감정을 인식할 얼굴을 흑백 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minSize=(30, 30))

    # 첫 번째 얼굴만 사용하여 사각형 그리기
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # 인식된 얼굴 영역이 있는 경우 감정 예측 수행
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # 이미지 정규화
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 모델로 감정 예측
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]  # 가장 높은 확률의 감정 라벨 선택
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면에 감정 인식 결과를 보여주기
    cv2.imshow('Emotion Detector', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 사용한 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()

