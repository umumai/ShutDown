# Middle finger and Pinkie Work


 #Import the libraries

import cv2
import mediapipe as mp
import subprocess #connect with terminal (Putting to sleep)
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
hands = mp_hands.Hands()

# Variables for gesture detection
last_gesture_time = 0
gesture_cooldown = 3  # 3 second cooldown between gestures 

def is_middle_finger_up(landmarks):
    """
    Check if middle finger is extended while other fingers are down
    """
    # Finger tip and pip landmarks
    # Thumb: 4, 3
    # Index: 8, 6
    # Middle: 12, 10
    # Ring: 16, 14
    # Pinky: 20, 18
    
    # Check if middle finger is up (tip higher than pip)
    middle_up = landmarks[12].y < landmarks[10].y
    
    # Check if other fingers are down (tip lower than pip)
    thumb_down = landmarks[4].x > landmarks[3].x  # For thumb, check x-axis
    index_down = landmarks[8].y > landmarks[6].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    
    return middle_up and thumb_down and index_down and ring_down and pinky_down

def is_pinkie_finger_up(landmarks):
    """
    Check if pinkie finger is extended while other fingers are down
    """
    # Check if pinkie finger is up (tip higher than pip)
    pinky_up = landmarks[20].y < landmarks[18].y
    
    # Check if other fingers are down (tip lower than pip)
    thumb_down = landmarks[4].x > landmarks[3].x  # For thumb, check x-axis
    index_down = landmarks[8].y > landmarks[6].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    
    return pinky_up and thumb_down and index_down and middle_down and ring_down

def put_computer_to_sleep():
    """
    Put the computer to sleep on macOS
    """
    try:
        # pmset command 
        subprocess.run(["pmset", "sleepnow"], check=True)
        print("Putting computer to sleep...")
        return True
    except subprocess.CalledProcessError:
        try:
            # Alt: use osascript
            subprocess.run(["osascript", "-e", "tell application \"System Events\" to sleep"], check=True)
            print("Putting computer to sleep...")
            return True
        except subprocess.CalledProcessError:
            print("Failed to put computer to sleep.")
            return False
while True:
    data, image = capture.read()
    # Flip the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # Storing the results
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Initialize status text
    gesture_status = "No hand detected"
    gesture_color = (0, 0, 255)  # Red
    
    # Check for middle finger gesture
    current_time = time.time()
    if results.multi_hand_landmarks:
        gesture_detected = False
        for hand_landmarks in results.multi_hand_landmarks:
            # Check if middle finger is up
            if is_middle_finger_up(hand_landmarks.landmark):
                gesture_detected = True
                gesture_status = "Middle finger detected!"
                gesture_color = (0, 255, 0)  # Green
                
                if current_time - last_gesture_time > gesture_cooldown:
                    last_gesture_time = current_time
                    cv2.putText(image, "Middle finger detected", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Wait 1 second then put computer to sleep
                    time.sleep(1)
                    put_computer_to_sleep()
                    #print("TEST: Middle finger gesture detected!")
                else:
                    remaining_cooldown = gesture_cooldown - (current_time - last_gesture_time)
                    gesture_status = f"Cooldown: {remaining_cooldown:.1f}s"
                    gesture_color = (0, 165, 255)  # Orange
            
            # Check if pinkie finger is up
            elif is_pinkie_finger_up(hand_landmarks.landmark):
                gesture_detected = True
                gesture_status = "Pinkie finger detected!"
                gesture_color = (0, 255, 0)  # Green
                
                if current_time - last_gesture_time > gesture_cooldown:
                    last_gesture_time = current_time
                    cv2.putText(image, "Pinkie finger detected", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Wait 1 second then put computer to sleep
                    time.sleep(1)
                    put_computer_to_sleep()
                    #print("TEST: Pinkie finger gesture detected!")
                else:
                    remaining_cooldown = gesture_cooldown - (current_time - last_gesture_time)
                    gesture_status = f"Cooldown: {remaining_cooldown:.1f}s"
                    gesture_color = (0, 165, 255)  # Orange
        
        if not gesture_detected:
            gesture_status = "Hand detected - No gesture"
            gesture_color = (255, 255, 0)  # Yellow
    
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    # Display gesture status
    cv2.putText(image, f"Status: {gesture_status}", (10, image.shape[0] - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    # Display instructions
    cv2.putText(image, "Show middle finger or pinkie to put computer to sleep", (10, image.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display quit instruction
    cv2.putText(image, "Press 'q' to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    # Display the image
    cv2.imshow('Hand Tracking', image)
    # Break the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
capture.release()
cv2.destroyAllWindows()
