from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load model
try:
    model = load_model('gender_detection.h5')  # Ensure this matches the model filename and format
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open video file
video_path = 'F:\!python\istockphoto-1196193908-640_adpp_is.mp4'  # Replace with your video file path
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Define class labels
classes = ['man', 'woman']

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop through frame
while video.isOpened():
    # Read frame from video
    status, frame = video.read()

    if not status:
        print("End of video or error reading frame.")
        break

    # Initialize counters for each frame
    men_count = 0
    women_count = 0

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (startX, startY, w, h) in faces:
        endX, endY = startX + w, startY + h

        # Draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = frame[startY:endY, startX:endX]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        confidence = conf[idx] * 100

        # Update the gender count
        if label == 'man':
            men_count += 1
        else:
            women_count += 1

        # Format label and confidence
        label = f"{label}: {confidence:.2f}%"

        # Write label and confidence above face rectangle
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display gender counts on the frame
    count_label = f"Men: {men_count} | Women: {women_count}"
    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display output
    cv2.imshow("Gender Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
