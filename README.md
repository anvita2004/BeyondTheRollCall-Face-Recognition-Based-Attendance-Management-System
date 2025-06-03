# BeyondTheRollCall-Face-Recognition-Based-Attendance-Management-System
PROJECT ON FACE RECOGINTION BASED ATTENDANCE MANAGEMENT SYSTEM WHICH IS CALLED "BEYOND THE ROLL CALL" FULL BASED ON PYTHON.

This project focuses on managing attendance through face recognition and can be used by any institution, school or office. It is a desktop application which does not require network.

DeepFace library is used for recognizing faces and FaceNet512 model is used. Model is trained on pre-captured images. This dataset is picked up from Kaggle. The dataset is augmented to create synthetic images to test the model. Since FaceNet512 is a highly accurate model, its accuracy came out to be 89%. Euclidean Distance is measured between detected faces and the embeddings, if the euclidean distance is less than a ceratin threshold then the face will be recognized with the name and roll no. If it more than the threshold decided then the face would be unknown.

Following are the libraries (dependencies) used:
 1. cv2 (OpenCV): Used for video capture, face detection, image manipulation, and displaying the webcam feed.
 2. pickle: Used to load and save serialized data (like embeddings, roll numbers, and names) to a file. This allows the system to persist face embeddings between sessions.
 3. pandas: Used for handling attendance data in the form of a DataFrame. It allows easy manipulation of data, such as adding attendance records, viewing them, and exporting them to a CSV file.
 4. datetime: Used to record the timestamp when a student’s attendance is marked. It helps format the current date and time when saving attendance.
 5. deepface: A deep learning library for face recognition. It provides pre-trained models and methods to extract face embeddings and compare them to identify known faces.
 6. numpy: Used for numerical operations like calculating the Euclidean distance between face embeddings to match faces. It handles arrays efficiently.
 7. os: Used to interact with the operating system. It helps check the existence of files and directories and manage file paths.
 8. tkinter: A Python library for creating graphical user interfaces (GUIs). It provides windows, buttons, labels, and other widgets to create interactive elements for the attendance system.
 9. PIL (Pillow): Used for image processing. In this case, it is used to resize and display background images in the Tkinter GUI.
 10. threading: Allows the webcam to run in a separate thread, enabling the GUI to remain responsive while the webcam is active.
 11. tkinter.filedialog: Used to open file dialog boxes, allowing the user to select a folder to save the exported attendance CSV file.

Following are the functionalities of the system:

1. Start camera: This button will start the webcam and detected faces will be recognized by the DeepFace and FaceNet512 showing their roll no and name. If the faces are not detected it will show unknown.
2. Stop camera: After recognizing face, to stop the camera, we have to trigger the button stop camera to stop the webcam.
3. View Attendance: This button will show us the attendance that could be daily, weekly or monthly along with the name, roll no and the timestamp at which the face was recognized.
4. Export Attendance: This will export the attendance recorded in a csv file format to any selected folder.
5. Clear Attendance: This button will simply clear all the recorded attendance.
6. Register New Face: This will register a new face. The person have to enter the roll no and name and then the webcam will be started to record the face so that the model can be trained on that face and can be converted to an embedding which would be saved in the pickle file. The person have to press "c" key to capture the face. After that, the face will be recognized when starting the camera.

OUTPUT-

Training the model:
![train 1](https://github.com/user-attachments/assets/4d2b4eb4-fa34-4d22-9a84-af1e80f30ecb)
![train 2](https://github.com/user-attachments/assets/b16f3701-8936-4658-9d7e-c88d623ada91)
![train 3](https://github.com/user-attachments/assets/6e75996f-f116-4f3d-87e6-c2f132ac97e7)

Testing the model:
![test 1](https://github.com/user-attachments/assets/a90cdf79-ff0f-46cd-a18c-17c62838543f)
![test 2](https://github.com/user-attachments/assets/b3026a7e-efe7-4fc3-8295-627489e01e57)
![test 3](https://github.com/user-attachments/assets/b4ea0fc0-f7c4-4c0a-9c06-e8f751dbb435)
![test 4](https://github.com/user-attachments/assets/2c4c2f46-86ac-4d4d-a20e-28511be34b3e)

Deploying the system in desktop using tkinter:
![Deploy 1](https://github.com/user-attachments/assets/75e948e4-ca4b-45e8-8a91-86936aecc00f)
![deploy 2](https://github.com/user-attachments/assets/388e380f-4b04-44bd-a317-e98e5af20204)
![deploy 3](https://github.com/user-attachments/assets/8bea09ec-c321-46b5-b749-08a26ebd749f)
![deploy 4](https://github.com/user-attachments/assets/c8e79a25-7e3c-406e-9978-144a76174b59)
![deploy 5](https://github.com/user-attachments/assets/40a550a1-0ded-491f-a05d-3e9e0000b484)




