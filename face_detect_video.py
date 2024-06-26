import cv2

face_cascade = cv2.CascadeClassifier('models\\haarcascade_frontalface_default.xml')
# Read the input video
cap = cv2.VideoCapture("C:\\Users\\Kaushal\\Desktop\\smaple.mp4")
# cap = cv2.VideoCapture()
while cap.isOpened():

    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
