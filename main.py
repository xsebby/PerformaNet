import cv2

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

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Optional: Add frame counter overlay
    frame_count += 1
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Camera Feed', frame)
    
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
cv2.destroyAllWindows()