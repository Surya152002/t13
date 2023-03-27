import cv2
import pytesseract
from flask import Flask, render_template, Response

app = Flask(__name__)

# set up the Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    # initialize the video capture device
    cap = cv2.VideoCapture(0)

    while True:
        # read the frame from the camera
        ret, frame = cap.read()

        # apply some image preprocessing to improve the OCR accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # detect the number plate in the frame using OpenCV
        plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # if a number plate is detected, extract the text using Tesseract OCR
        if len(plates) > 0:
            for (x, y, w, h) in plates:
                plate_img = gray[y:y+h, x:x+w]
                plate_text = pytesseract.image_to_string(plate_img, config='--psm 11')
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # convert the frame to a JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
