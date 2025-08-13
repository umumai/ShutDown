import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)

# Landmark indices for finger tips and pips (fixed)
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False
    if results.multi_hand_landmarks and results.multi_handedness:
        h, w, _ = frame.shape
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_detected = True
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            landmarks = hand_landmarks.landmark
            # Count fingers for this hand
            finger_count = 0
            if landmarks[FINGER_TIPS[0]].x > landmarks[FINGER_PIPS[0]].x:
                finger_count += 1
            for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
                if landmarks[tip].y < landmarks[pip].y:
                    finger_count += 1
            # Set color based on finger count
            if finger_count == 0:
                box_color = (0, 0, 255)  # Red
            elif finger_count == 1:
                box_color = (0, 165, 255)  # Orange
            elif finger_count == 2:
                box_color = (0, 255, 255)  # Yellow
            elif finger_count == 3:
                box_color = (0, 255, 0)    # Green
            elif finger_count == 4:
                box_color = (255, 0, 0)    # Blue
            else:
                box_color = (255, 0, 255)  # Magenta
            # Calculate bounding box for the hand
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # Get handedness label
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            # Draw tracking box, finger count, and handedness for this hand
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), box_color, 4)
            cv2.putText(frame, f'{hand_label} Hand', (x_min, y_min-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2)
            cv2.putText(frame, f'Fingers: {finger_count}', (x_min, y_min-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)
    if not hand_detected:
        # fallback: draw static box if no hand
        cv2.rectangle(frame, (20, 20), (170, 120), (0,0,255), -1)
        cv2.putText(frame, f'Fingers: 0', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

    cv2.imshow('Finger Counter', frame)
    if cv2.waitKey(5) & 0xFF == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()
