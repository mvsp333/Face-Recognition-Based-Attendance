# import streamlit as st
# import cv2
# import numpy as np
# import face_recognition
# import os
# import csv
# from datetime import datetime

# # Set the path for storing images
# path = "attendance-images"
# images = []
# studentIDs = []
# studentList = os.listdir(path)
# current_student = None

# # Load images and student IDs
# for sid in studentList:
#     curImg = cv2.imread(f"{path}/{sid}")
#     images.append(curImg)
#     studentIDs.append(os.path.splitext(sid)[0])

# class Student:
#     def __init__(self, name):
#         self.name = name
#         self.attended = False

#     def markAsAttended(self):
#         if self.attended:
#             return "Already Marked"
#         else:
#             now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             with open("Attendance-List.csv", "a+", newline="") as f:
#                 writer = csv.writer(f)
#                 f.seek(0)
#                 studentDataList = [line for line in csv.reader(f)]
#                 student_names = [row[1] for row in studentDataList]

#                 if self.name in student_names:
#                     self.attended = True
#                     return "Already Marked"
#                 else:
#                     writer.writerow(
#                         [len(studentDataList) + 1, self.name, "Present", now]
#                     )
#                     self.attended = True
#                     return "Marked"

# def findFaceEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faceEncode = face_recognition.face_encodings(img)[0]
#         encodeList.append(faceEncode)
#     return encodeList

# knownStudentFaceEncodings = findFaceEncodings(images)
# cap = cv2.VideoCapture(0)

# students = []
# for name in studentIDs:
#     students.append(Student(name.upper()))

# def mark_attendance_frame(frame):
#     global current_student
#     imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#     facesCurFrame = face_recognition.face_locations(imgS)
#     faceEncodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

#     for faceEncode, faceLoc in zip(faceEncodingsCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(knownStudentFaceEncodings, faceEncode)
#         faceDis = face_recognition.face_distance(knownStudentFaceEncodings, faceEncode)
#         matchIndex = np.argmin(faceDis)
#         if matches[matchIndex]:
#             student = students[matchIndex]
#             current_student = student.name
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#             color = (0, 255, 0)
#             if student.attended:
#                 color = (0, 0, 255)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
#             cv2.putText(
#                 frame,
#                 student.name,
#                 (x1 + 6, y2 - 6),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1,
#                 (255, 255, 255),
#                 2,
#             )
#             student.markAsAttended()
#     return frame

# def video_capture():
#     ret, frame = cap.read()
#     if ret:
#         frame = mark_attendance_frame(frame)
#         return frame
#     return None

# # Streamlit UI
# st.title("Attendance System")

# st.sidebar.title("Navigation")
# options = st.sidebar.radio("Select Page", ["Home", "Register", "Attendance", "Attended Students"])

# if options == "Home":
#     st.header("Welcome to the Attendance System")
#     # st.image("static/home_image.jpg")

# elif options == "Register":
#     st.header("Register a New Student")
#     name = st.text_input("Name")
#     student_id = st.text_input("Student ID")
#     image = st.file_uploader("Upload Student Image", type=["jpg", "png", "jpeg"])

#     if st.button("Register"):
#         if student_id in studentIDs:
#             st.error("Student ID already exists!")
#         elif image is not None:
#             image_path = os.path.join(path, f"{student_id}.jpg")
#             with open(image_path, "wb") as f:
#                 f.write(image.getbuffer())
#             studentIDs.append(student_id)
#             images.append(cv2.imread(image_path))
#             students.append(Student(name.upper()))
#             knownStudentFaceEncodings = findFaceEncodings(images)
#             st.success("Student registered successfully!")
#         else:
#             st.error("Error registering student. Please try again.")

# elif options == "Attendance":
#     st.header("Attendance")
#     st.write("Current student:", current_student)
#     frame_placeholder = st.empty()
    
#     while True:
#         frame = video_capture()
#         if frame is not None:
#             frame_placeholder.image(frame, channels="BGR")
#         if st.button("Stop"):
#             break

# elif options == "Attended Students":
#     st.header("Attended Students")
#     with open("Attendance-List.csv", "r") as f:
#         reader = csv.reader(f)
#         attended_students = list(reader)
#     for student in attended_students:
#         st.write(student)

# # Run Streamlit app
# if __name__ == "__main__":
#     st.write("Running the Streamlit app")




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


knownStudentFaceEncodings = findFaceEncodings(images)
cap = cv2.VideoCapture(0)

students = []
for name in studentIDs:
    students.append(Student(name.upper()))


def gen_frames():
    global current_student
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
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
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
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


