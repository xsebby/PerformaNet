import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_holding_cup(landmarks):
    """
    Detect if hand is holding a cup/matcha based on hand landmarks.
    A holding gesture typically has:
    - Fingers curled (tips closer to palm)
    - Thumb positioned to grip
    - Hand forming a C-shape or grip shape
    """
    # Get key landmarks
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    
    # Check if fingers are curled (tip is closer to wrist than pip joint)
    def is_finger_curled(tip, pip, wrist):
        tip_to_wrist = calculate_distance(tip, wrist)
        pip_to_wrist = calculate_distance(pip, wrist)
        return tip_to_wrist < pip_to_wrist * 1.2  # Allow some tolerance
    
    # Check if thumb is positioned for gripping (thumb tip is closer to fingers)
    thumb_to_index = calculate_distance(thumb_tip, index_tip)
    thumb_to_wrist = calculate_distance(thumb_tip, wrist)
    
    # Count how many fingers are curled
    curled_fingers = 0
    if is_finger_curled(index_tip, index_pip, wrist):
        curled_fingers += 1
    if is_finger_curled(middle_tip, middle_pip, wrist):
        curled_fingers += 1
    if is_finger_curled(ring_tip, ring_pip, wrist):
        curled_fingers += 1
    if is_finger_curled(pinky_tip, pinky_pip, wrist):
        curled_fingers += 1
    
    # Thumb should be positioned to grip (not fully extended)
    thumb_curled = calculate_distance(thumb_tip, thumb_mcp) < calculate_distance(thumb_mcp, wrist) * 0.8
    
    # Hand is holding if:
    # - At least 3 fingers are curled (forming a grip)
    # - Thumb is positioned for gripping
    # - Thumb is relatively close to index finger (forming a C-shape)
    is_holding = (
        curled_fingers >= 3 and
        thumb_curled and
        thumb_to_index < thumb_to_wrist * 0.6
    )
    
    return is_holding

# Initialize camera
cap = cv2.VideoCapture(0)

# Set camera properties (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully")
print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print("\nControls:")
print("  q - Quit")
print("  s - Save screenshot")
print("\nDetecting performative gestures (holding cup/matcha)...")

frame_count = 0
performative_detected = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Reset performative flag
    performative_detected = False
    
    # Draw hand landmarks and detect holding gesture
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Check if hand is holding cup/matcha
            if is_holding_cup(hand_landmarks.landmark):
                performative_detected = True
    
    # Add frame counter overlay
    frame_count += 1
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display performative status
    if performative_detected:
        cv2.putText(frame, "PERFORMATIVE: Holding Cup/Matcha!", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # Draw a border around the frame to highlight detection
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 255), 5)
    
    # Display the frame
    cv2.imshow('Camera Feed - PerformaNet', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Quit on 'q'
    if key == ord('q'):
        break
    # Save screenshot on 's'
    elif key == ord('s'):
        filename = f"screenshot_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")

cap.release()
hands.close()
cv2.destroyAllWindows()