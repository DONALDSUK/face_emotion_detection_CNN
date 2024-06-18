import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n-face.pt')  # GPU 사용을 가정

# 웹캠에서 비디오 캡쳐 객체 생성
cap = cv2.VideoCapture(0)  # 0번 웹캠 사용

# 웹캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 웹캠에서 프레임을 계속 읽기
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLOv8을 이용해 프레임에서 얼굴 검출
    results = model.predict(frame)

    # 검출된 얼굴을 프레임에서 자르기
    cropped_faces = []
    for result in results:
        boxes = result.boxes.xyxy  # 검출된 박스의 좌표들
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = frame[y1:y2, x1:x2]
            cropped_faces.append(crop_img)

            # 자른 이미지 보여주기 (새 창에서 각 얼굴을 따로 보여줄 수 있음)
            cv2.imshow(f'Cropped Face', crop_img)

    # 처리된 전체 프레임 보여주기
    cv2.imshow('Webcam Face Detection', frame)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
