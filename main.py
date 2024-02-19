import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils


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
    grab_threshold = 50  # Define a threshold for proximity to the object

    # Calculate the center of the bounding box of the object
    x, y, w, h = bbox
    object_center = (x + w // 2, y + h // 2)

    # Count how many fingertips are close to the object
    close_fingertips = 0
    for fingertip in fingertips:
        fingertip_pos = hand_landmarks[fingertip]
        distance = np.sqrt(
            (fingertip_pos[0] - object_center[0]) ** 2
            + (fingertip_pos[1] - object_center[1]) ** 2
        )
        if distance < grab_threshold:
            close_fingertips += 1

    # Assuming a grab if at least 3 fingertips are close to the object center
    return close_fingertips >= 3


def detect_cube_like_objects(frame, hand_landmarks):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_detected = False
    action_text = ""

    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(cnt) > 400:  # Detect cube-like objects
            x, y, w, h = cv2.boundingRect(approx)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)

            # Check for grabbing
            if hand_landmarks and is_grabbing(hand_landmarks, (x, y, w, h)):
                action_detected = True
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

            # Check if any hand landmark is inside the object's bounding box
            elif hand_landmarks:
                for point in hand_landmarks:
                    if x < point[0] < x + w and y < point[1] < y + h:
                        action_detected = True
                        action_text = "Touch Detected!"
                        cv2.putText(
                            frame,
                            action_text,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )
                        break

    return frame, action_detected


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, hand_landmarks = detect_hand(frame)
    frame, touch_detected = detect_cube_like_objects(frame, hand_landmarks)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
