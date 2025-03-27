import cv2

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Check if the cascades were loaded successfully
if face_cascade.empty():
    print("Error loading face cascade!")
    exit()
if eye_cascade.empty():
    print("Error loading eye cascade!")
    exit()

# Initialize video capture from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video capture!")
    exit()

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        print("No frame captured! Exiting...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Extract the region of interest (ROI) for the face
        face_roi = gray[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the detected eyes
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

    # Display the frame with detected faces and eyes
    cv2.imshow('Capture - Face Detection', frame)

    # Exit the loop if the 'ESC' key is pressed
    if cv2.waitKey(10) == 27:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()