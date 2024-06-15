import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array


face_classifier = cv2.CascadeClassifier(
    "D:\\College\\facial-emotion-recorder\\models\\haarcascade_frontalface_default.xml"
)

classifier = load_model("D:/College/facial-emotion-recorder/models/model.h5")


emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

emotion_counts = {emotion: 0 for emotion in emotion_labels}
total_emotion_counts=0

cap = cv2.VideoCapture("C:\\Users\\Kaushal\\Desktop\\smaple.mp4")
# cap = cv2.VideoCapture(0)

# Make the window resizable
cv2.namedWindow("Emotion Detector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Emotion Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            total_emotion_counts+=1
            emotion_counts[label]+=1
            label_position = (x, y)
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame, "No Faces", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

    # Combine emotion counts into a single string
    emotion_text = f"Total ({total_emotion_counts}): "
    for emotion, count in emotion_counts.items():
        emotion_text += f"{emotion}({count}) "

    cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'r' to reset and 'q' to exit", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Emotion Detector", frame)

    key = cv2.waitKey(1) & 0xFF  # Get the pressed key without extended info
    if key == ord("r"):
        emotion_counts = {emotion: 0 for emotion in emotion_labels}
        total_emotion_counts=0
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
