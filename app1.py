from flask import Flask, render_template, request, flash, Response, session, jsonify
import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"

path = "attendance-images"
images = []
studentIDs = []
studentList = os.listdir(path)
current_student = None

app.static_folder = "static"

for sid in studentList:
    curImg = cv2.imread(f"{path}/{sid}")
    images.append(curImg)
    studentIDs.append(os.path.splitext(sid)[0])

class Student:
    def __init__(self, name):
        self.name = name
        self.attended = False

    def markAsAttended(self):
        if self.attended:
            return "Already Marked"
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("Attendance-List.csv", "a+", newline="") as f:
                writer = csv.writer(f)
                f.seek(0)
                studentDataList = [line for line in csv.reader(f)]
                student_names = [row[1] for row in studentDataList]

                if self.name in student_names:
                    self.attended = True
                    return "Already Marked"
                else:
                    writer.writerow(
                        [len(studentDataList) + 1, self.name, "Present", now]
                    )
                    self.attended = True
                    return "Marked"

def findFaceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceEncode = face_recognition.face_encodings(img)[0]
        encodeList.append(faceEncode)
    return encodeList

# # Use raw strings to avoid unicode escape error
# weightsPath = r"C:\Users\Dell\OneDrive\Desktop\mini projects\Face Recognition Attendance System-20240721T140441Z-001\Face Recognition Attendance System\yolov7\yolov7.weights"
# configPath = r"C:\Users\Dell\OneDrive\Desktop\mini projects\Face Recognition Attendance System-20240721T140441Z-001\Face Recognition Attendance System\yolov7\cfg\traning\yolov7.cfg"

# # Use absolute paths to avoid path issues
# weightsPath = os.path.abspath(weightsPath)
# configPath = os.path.abspath(configPath)

# print(f"Using weights file: {weightsPath}")
# print(f"Using config file: {configPath}")

# # Load YOLO model
# try:
#     yolo_net = cv2.dnn.readNet(weightsPath, configPath)
#     print("YOLO model loaded successfully.")
# except cv2.error as e:
#     print(f"Error loading YOLO model: {e}")

knownStudentFaceEncodings = findFaceEncodings(images)
cap = cv2.VideoCapture(0)

students = []
for name in studentIDs:
    students.append(Student(name.upper()))

# Load YOLOv7
yolo_net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]


def detect_faces_yolo(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = scores[np.argmax(scores)]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    final_boxes = []
    for i in indices:
        i = i[0]
        final_boxes.append(boxes[i])
    return final_boxes


def gen_frames():
    global current_student
    while True:
        success, img = cap.read()
        faces = detect_faces_yolo(img)
        faceEncodingsCurFrame = []
        face_locations = []

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face_img)
            if encodings:
                faceEncodingsCurFrame.append(encodings[0])
                face_locations.append((y, x+w, y+h, x))

        for faceEncode, faceLoc in zip(faceEncodingsCurFrame, face_locations):
            matches = face_recognition.compare_faces(
                knownStudentFaceEncodings, faceEncode
            )
            faceDis = face_recognition.face_distance(
                knownStudentFaceEncodings, faceEncode
            )
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                student = students[matchIndex]
                current_student = student.name
                y1, x2, y2, x1 = faceLoc
                color = (0, 255, 0)
                if student.attended:
                    color = (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(
                    img,
                    student.name,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

        ret, buffer = cv2.imencode(".jpg", img)
        frame = buffer.tobytes()
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        student_id = request.form["student_id"]

        if student_id in studentIDs:
            flash("Student ID already exists!", "danger")
            return render_template("register.html")

        if "image" in request.files:
            image = request.files["image"]
            if image.filename != "":
                image_path = os.path.join(path, f"{name}.jpg")
                image.save(image_path)
                studentIDs.append(student_id)
                images.append(cv2.imread(image_path))
                students.append(Student(name.upper()))
                global knownStudentFaceEncodings
                knownStudentFaceEncodings = findFaceEncodings(images)
                flash("Student registered successfully!", "success")
                return render_template("index.html")

        flash("Error registering student. Please try again.", "danger")
        return render_template("register.html")

    return render_template("register.html")

@app.route("/attendance")
def attendance():
    return render_template("attendance.html", current_student=current_student)

@app.route("/capture", methods=["POST"])
def capture():
    global current_student
    ret, frame = cap.read()
    if ret:
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        faceEncodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for faceEncode, faceLoc in zip(faceEncodingsCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(
                knownStudentFaceEncodings, faceEncode
            )
            faceDis = face_recognition.face_distance(
                knownStudentFaceEncodings, faceEncode
            )
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                student = students[matchIndex]
                current_student = student.name
                result = student.markAsAttended()
                return jsonify(
                    {
                        "status": "success",
                        "message": f"Student identified: {student.name}. Status: {result}",
                    }
                )

        return jsonify(
            {
                "status": "fail",
                "message": "No known student identified in the captured image.",
            }
        )
    else:
        return jsonify(
            {"status": "error", "message": "Error capturing image. Please try again."}
        )

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == "admin" and check_password_hash(
            generate_password_hash("admin"), password
        ):
            session["user"] = username
            flash("Logged in successfully.", "success")
            return render_template("attended_students.html")
        else:
            flash("Invalid username or password", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return render_template("index.html")

@app.route("/attended_students")
def attended_students_list():
    if "user" not in session:
        flash("Please log in to access this page.", "warning")
        return render_template("login.html")

    attended_students = []
    with open("Attendance-List.csv", "r") as f:
        reader = csv.reader(f)
        attended_students = list(reader)

    return render_template(
        "attended_students.html",
        attended_students=attended_students,
        current_student=current_student,
    )

@app.route("/access_denied")
def access_denied():
    return render_template("access_denied.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=True)
