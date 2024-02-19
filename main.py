import cv2
import mediapipe as mp
import numpy as np
import socket
import torch


# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


mp_drawing = mp.solutions.drawing_utils

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def detect_objects_yolo(frame, hand_landmarks):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize and normalize image for YOLOv5
    results = model(frame_rgb)

    # Process results
    detections = results.pandas().xyxy[0]  # DataFrame of detections
    for i, det in detections.iterrows():
        xmin, ymin, xmax, ymax = (
            int(det["xmin"]),
            int(det["ymin"]),
            int(det["xmax"]),
            int(det["ymax"]),
        )
        cv2.rectangle(
            frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2
        )  # Draw rectangle around detected objects

        if hand_landmarks and is_grabbing(hand_landmarks, (xmin, ymin, xmax, ymax)):
            action_text = "Grab Detected!"
            cv2.putText(
                frame,
                action_text,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

    return frame, detections


def detect_hand(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_landmarks = []

    # Draw hand landmarks and store their positions
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmark.landmark:
                # Convert landmark position to relative pixel positions
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                hand_landmarks.append((x, y))

    return frame, hand_landmarks


def is_grabbing(hand_landmarks, bbox):
    fingertips = [
        4,
        8,
        12,
        16,
        20,
    ]  # Indices for fingertips in MediaPipe hand landmarks
    grab_threshold = 300  # Define a threshold for proximity to the object

    # Calculate the center of the bounding box of the object
    x, y, w, h = bbox
    object_center = ((x + w) // 2, (y + h) // 2)

    cv2.rectangle(frame, object_center, object_center, (0, 255, 0), 100)

    # Count how many fingertips are close to the object
    close_fingertips = 0
    for fingertip in fingertips:
        fingertip_pos = hand_landmarks[fingertip]
        distance = np.sqrt(
            (fingertip_pos[0] - object_center[0]) ** 2
            + (fingertip_pos[1] - object_center[1]) ** 2
        )
        print(distance)
        if distance < grab_threshold:
            close_fingertips += 1

    # Assuming a grab if at least 3 fingertips are close to the object center
    return close_fingertips >= 3


cap = cv2.VideoCapture(1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, hand_landmarks = detect_hand(frame)
    frame, detections = detect_objects_yolo(
        frame, hand_landmarks
    )  # Replace or combine with your cube detection logic

    # if hand_landmarks and not detections.empty:
    #     detect_grabbing(frame, detections, hand_landmarks)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
