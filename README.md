# facial-emotion-recorder
Hello, I started this project to log the emotions of the people over a period of time. This project demonstrates real-time or video-source facial emotion detection using OpenCV, Keras, and a pre-trained deep learning model.

## Dependencies
- haarcascade_frontalface_default CascadeClassifier
- pre-trained model for emotion detection (model.h5)
- Python libraries (OpenCV, Tensorflow, NumPY)

## Technical Skills
- Python programming
- OpenCV for computer vision tasks (face detection)
- Keras for deep learning model loading and prediction
- NumPy for image processing

## Working flow
- Captures video from a webcam or processes a video file.
- Detects faces in each frame using the Haar cascade classifier.
- Predicts emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) for each detected face using the pre-trained Keras model.
- Displays bounding boxes around detected faces.
- Overlays predicted emotions on top of the faces.
- Tracks and displays total and individual emotion counts throughout the video.
- Allows resetting emotion counts with the 'r' key and exiting with 'q'.

## Future possibilities
- Market research: Detailed insights while shopping in malls.
- Mental health: personal mood and emotion tracking.
- More engaging software solutions.

