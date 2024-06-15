import cv2

face_cascade = cv2.CascadeClassifier('models\\haarcascade_frontalface_default.xml')

img_file_path = 'C:\\Users\\Kaushal\\Pictures\\Camera Roll\\WIN_20240323_11_37_35_Pro.jpg'
# img_file_path = 'D:\\College\\facial-emotion-recorder\\testing\\sampleimg.jpg'
# Read the input image
img = cv2.imread(img_file_path)

# Check if image is loaded successfully
if img is None:
    print("Error: Could not read image")
else:
    # Process the image selected
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)

    # Display the output
    cv2.imshow('img', img)
    cv2.waitKey(0)  # Press any key to exit
    cv2.destroyAllWindows()
