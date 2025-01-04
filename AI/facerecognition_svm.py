import face_recognition
from sklearn import svm
import os
import cv2
import datetime

# Fungsi untuk melatih SVC classifier
def train_svc(train_dir):
    encodings = []
    names = []

    # Loop melalui setiap orang dalam direktori pelatihan
    for person in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, person)
        if not os.path.isdir(person_dir):
            continue

        # Loop melalui setiap gambar pelatihan untuk orang saat ini
        for person_img in os.listdir(person_dir):
            img_path = os.path.join(person_dir, person_img)
            face = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(face)

            # Jika gambar pelatihan mengandung tepat satu wajah
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                encodings.append(face_enc)
                names.append(person)
            else:
                print(f"{img_path} was skipped and can't be used for training")

    # Buat dan latih SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings, names)
    return clf

# Fungsi untuk memprediksi wajah dalam gambar
def predict_faces(frame, clf):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    predictions = []

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_enc = face_recognition.face_encodings(rgb_frame, [face_location])[0]
        name = clf.predict([face_enc])
        predictions.append((name[0], (top, right, bottom, left)))

    return predictions

# Fungsi untuk menampilkan hasil prediksi di frame
def show_predictions(frame, predictions, logged_names, log_file_name):
    for name, (top, right, bottom, left) in predictions:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Catat nama ke file log jika belum pernah dicatat
        if name not in logged_names:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file_name, "a") as log_file:
                log_file.write(f"{name}, {timestamp}\n")
            logged_names.add(name)

if __name__ == "__main__":
    # Latih SVC classifier
    train_dir = 'C:\\Brandon Li\\AOL_AI\\AI\\svm_examples\\train'
    clf = train_svc(train_dir)
    print("Training complete!")

    log_file_name = "Face_Attendance_" + datetime.datetime.now().strftime("%Y-%m-%d") + ".txt"
    logged_names = set()  # Set untuk menyimpan nama yang sudah dicatat

    # Buka kamera untuk pengenalan wajah real-time
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        # Prediksi wajah dalam frame
        predictions = predict_faces(frame, clf)

        # Tampilkan hasil prediksi di frame
        show_predictions(frame, predictions, logged_names, log_file_name)

        # Tampilkan frame
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# import face_recognition
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# import os
# import numpy as np
# import time

# def load_images_from_folder(folder):
#     encodings = []
#     labels = []
#     for person_name in os.listdir(folder):
#         person_folder = os.path.join(folder, person_name)
#         if not os.path.isdir(person_folder):
#             continue
#         for image_name in os.listdir(person_folder):
#             image_path = os.path.join(person_folder, image_name)
#             if not os.path.isfile(image_path):
#                 continue
#             image = face_recognition.load_image_file(image_path)
#             face_bounding_boxes = face_recognition.face_locations(image)
#             if len(face_bounding_boxes) != 1:
#                 continue
#             face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
#             encodings.append(face_encoding)
#             labels.append(person_name)
#     return encodings, labels

# # Load training data
# train_dir = 'C:\\Brandon Li\\AOL_AI\\AI\\svm_examples\\train'
# X_train, y_train = load_images_from_folder(train_dir)

# # Check if training data is loaded correctly
# if len(X_train) == 0 or len(y_train) == 0:
#     raise ValueError("No training data found. Please check the training directory.")

# # Load testing data
# test_dir = 'C:\\Brandon Li\\AOL_AI\\AI\\svm_examples\\test'
# X_test, y_test = load_images_from_folder(test_dir)

# # Check if testing data is loaded correctly
# if len(X_test) == 0 or len(y_test) == 0:
#     raise ValueError("No testing data found. Please check the testing directory.")

# # Convert lists to numpy arrays
# X_train = np.array(X_train)
# X_test = np.array(X_test)

# # Train the SVM classifier
# clf = svm.SVC(kernel='linear', class_weight='balanced')

# start_time = time.time()
# clf.fit(X_train, y_train)
# end_time = time.time()

# training_time = end_time - start_time
# print(f"Training complete! Time taken: {training_time:.2f} seconds")

# # Predict the labels for the test set
# start_time = time.time()
# y_pred = clf.predict(X_test)
# end_time = time.time()
# prediction_time = end_time - start_time
# print(f"Prediction complete! Time taken: {prediction_time:.2f} seconds")

# # Calculate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")
