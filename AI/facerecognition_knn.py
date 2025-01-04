# import datetime
# import os
# import pickle
# from PIL import Image, ImageDraw
# import face_recognition
# from sklearn import neighbors
# import cv2
# import numpy as np

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# base_dir = "knn_examples"
# train_dir = os.path.join(base_dir, "train")
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)

# logged_names = set()
# last_log_time = None
# log_file_path = None

# # Buat file log baru setiap 2 jam
# def get_log_file():
#     global last_log_time, log_file_path, logged_names
#     current_time = datetime.datetime.now()
#     if not last_log_time or (current_time - last_log_time).total_seconds() >= 2 * 3600:
#         last_log_time = current_time
#         log_file_name = current_time.strftime("Face_Attendance_%Y-%m-%d_%H-%M.txt")
#         log_file_path = os.path.join(log_dir, log_file_name)
#         logged_names.clear()  # Reset nama yang sudah dicatat
#     return log_file_path

# # Fungsi logging nama ke file
# def log_recognition(name):
#     if name not in logged_names:
#         log_file = get_log_file()
#         timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(log_file, "a") as f:
#             f.write(f"{name}, {timestamp}\n")
#         logged_names.add(name)

# # Fungsi pelatihan model KNN
# def train(train_dir, model_save_path="trained_knn_model.clf", n_neighbors=5):
#     if os.path.exists(model_save_path):
#         print("Model found, skipping training.")
#         with open(model_save_path, 'rb') as f:
#             return pickle.load(f)

#     X, y = [], []
#     for class_dir in os.listdir(train_dir):
#         class_path = os.path.join(train_dir, class_dir)
#         if not os.path.isdir(class_path):
#             continue
#         for img_path in os.listdir(class_path):
#             full_path = os.path.join(class_path, img_path)
#             image = face_recognition.load_image_file(full_path)
#             face_bounding_boxes = face_recognition.face_locations(image)
#             if len(face_bounding_boxes) == 1:
#                 X.append(face_recognition.face_encodings(image, face_bounding_boxes)[0])
#                 y.append(class_dir)

#     if not X:
#         raise ValueError("No valid training data found.")

#     if n_neighbors is None:
#         n_neighbors = int(round(len(X)**0.5))
#     knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
#     knn_clf.fit(X, y)

#     with open(model_save_path, 'wb') as f:
#         pickle.dump(knn_clf, f)

#     print("Training complete.")
#     return knn_clf

# def predict(rgb_image, knn_clf, distance_threshold=0.6):
#     face_locations = face_recognition.face_locations(rgb_image)
#     if not face_locations:
#         return []
    
#     face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
#     closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
#     are_matches = [dist[0] <= distance_threshold for dist in closest_distances[0]]
    
#     # Di sini, jika wajah tidak cocok dengan data pelatihan, beri label "unknown"
#     return [
#         (name, loc) if match else ("unknown", loc)
#         for name, loc, match in zip(knn_clf.predict(face_encodings), face_locations, are_matches)
#     ]

# # Tampilkan prediksi di video
# def show_predictions_on_frame(frame, predictions):
#     pil_image = Image.fromarray(frame)
#     draw = ImageDraw.Draw(pil_image)
#     for name, (top, right, bottom, left) in predictions:
#         color = (0, 255, 0) if name != "unknown" else (255, 0, 0)
#         draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)
#         draw.text((left, bottom + 5), name, fill=color)
#         if name != "unknown":
#             log_recognition(name)
#     return np.array(pil_image)

# # Fungsi utama untuk menangkap video
# def run_camera(knn_clf):
#     video_capture = cv2.VideoCapture(0)
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         predictions = predict(rgb_frame, knn_clf)
#         output_frame = show_predictions_on_frame(frame, predictions)
#         cv2.imshow("Video", output_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video_capture.release()
#     cv2.destroyAllWindows()

# # Main script
# if __name__ == "__main__":
#     print("Loading KNN model...")
#     knn_clf = train(train_dir)
#     print("Starting camera...")
#     run_camera(knn_clf)

# import face_recognition
# from sklearn import neighbors
# from sklearn.metrics import accuracy_score
# import os
# import numpy as np
# import time
# import pickle

# def load_images_from_folder(folder):
#     encodings = []
#     labels = []
#     for person_name in os.listdir(folder):
#         person_folder = os.path.join(folder, person_name)
#         if not os.path.isdir(person_folder):
#             continue
#         for image_name in os.listdir(person_folder):
#             image_path = os.path.join(person_folder, image_name)
#             image = face_recognition.load_image_file(image_path)
#             face_bounding_boxes = face_recognition.face_locations(image)
#             if len(face_bounding_boxes) != 1:
#                 continue
#             face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
#             encodings.append(face_encoding)
#             labels.append(person_name)
#     return encodings, labels

# # Load training data
# train_dir = 'C:\\Brandon Li\\AOL_AI\\AI\\knn_examples\\train'
# start_time = time.time()
# X_train, y_train = load_images_from_folder(train_dir)
# end_time = time.time()
# loading_time_train = end_time - start_time
# print(f"Loading training data complete! Time taken: {loading_time_train:.2f} seconds")

# # Load testing data
# test_dir = 'C:\\Brandon Li\\AOL_AI\\AI\\knn_examples\\test'
# start_time = time.time()
# X_test, y_test = load_images_from_folder(test_dir)
# end_time = time.time()
# loading_time_test = end_time - start_time
# print(f"Loading testing data complete! Time taken: {loading_time_test:.2f} seconds")

# # Train the KNN classifier
# knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')

# start_time = time.time()
# knn_clf.fit(X_train, y_train)
# end_time = time.time()
# training_time = end_time - start_time
# print(f"Training complete! Time taken: {training_time:.2f} seconds")

# # Save the model
# model_save_path = "trained_knn_model.clf"
# with open(model_save_path, 'wb') as f:
#     pickle.dump(knn_clf, f)

# # Predict the labels for the test set
# start_time = time.time()
# y_pred = knn_clf.predict(X_test)
# end_time = time.time()
# prediction_time = end_time - start_time
# print(f"Prediction complete! Time taken: {prediction_time:.2f} seconds")

# # Calculate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

import datetime
import os
import pickle
from PIL import Image, ImageDraw
import face_recognition
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

base_dir = "knn_examples"
train_dir = os.path.join(base_dir, "train")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logged_names = set()
last_log_time = None
log_file_path = None

# Buat file log baru setiap 2 jam
def get_log_file():
    global last_log_time, log_file_path, logged_names
    current_time = datetime.datetime.now()
    if not last_log_time or (current_time - last_log_time).total_seconds() >= 2 * 3600:
        last_log_time = current_time
        log_file_name = current_time.strftime("Face_Attendance_%Y-%m-%d_%H-%M.txt")
        log_file_path = os.path.join(log_dir, log_file_name)
        logged_names.clear()  # Reset nama yang sudah dicatat
    return log_file_path

# Fungsi logging nama ke file
def log_recognition(name):
    if name not in logged_names:
        log_file = get_log_file()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{name}, {timestamp}\n")
        logged_names.add(name)

# Fungsi untuk melatih dan menguji akurasi dengan berbagai nilai k
def train_and_evaluate_k_values(train_dir, k_values=[1, 3, 5, 7, 9]):
    X, y = [], []
    
    # Ambil data pelatihan dan encoding wajah
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_path in os.listdir(class_path):
            full_path = os.path.join(class_path, img_path)
            image = face_recognition.load_image_file(full_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) == 1:
                X.append(face_recognition.face_encodings(image, face_bounding_boxes)[0])
                y.append(class_dir)

    if not X:
        raise ValueError("No valid training data found.")

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    accuracies = {}

    for k in k_values:
        print(f"Training with k = {k}...")
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', weights='distance')
        knn_clf.fit(X_train, y_train)

        # Prediksi pada data uji
        y_pred = knn_clf.predict(X_test)
        
        # Hitung akurasi
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[k] = accuracy
        print(f"Accuracy for k={k}: {accuracy * 100:.2f}%")

    return accuracies

def predict(rgb_image, knn_clf, distance_threshold=0.6):
    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return []
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [dist[0] <= distance_threshold for dist in closest_distances[0]]
    
    # Di sini, jika wajah tidak cocok dengan data pelatihan, beri label "unknown"
    return [
        (name, loc) if match else ("unknown", loc)
        for name, loc, match in zip(knn_clf.predict(face_encodings), face_locations, are_matches)
    ]

# Tampilkan prediksi di video
def show_predictions_on_frame(frame, predictions):
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in predictions:
        color = (0, 255, 0) if name != "unknown" else (255, 0, 0)
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)
        draw.text((left, bottom + 5), name, fill=color)
        if name != "unknown":
            log_recognition(name)
    return np.array(pil_image)

# Fungsi utama untuk menangkap video
def run_camera(knn_clf):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predict(rgb_frame, knn_clf)
        output_frame = show_predictions_on_frame(frame, predictions)
        cv2.imshow("Video", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    print("Evaluating KNN model with different values of k...")
    accuracies = train_and_evaluate_k_values(train_dir)
    print("\nAkurasi untuk setiap k:")
    for k, accuracy in accuracies.items():
        print(f"k={k}: {accuracy * 100:.2f}%")
    
    # Melanjutkan dengan pemilihan k terbaik dan penggunaan model
    best_k = max(accuracies, key=accuracies.get)
    print(f"Model terbaik dengan k={best_k}")
    
    # Latih model menggunakan k terbaik
    X, y = [], []
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_path in os.listdir(class_path):
            full_path = os.path.join(class_path, img_path)
            image = face_recognition.load_image_file(full_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) == 1:
                X.append(face_recognition.face_encodings(image, face_bounding_boxes)[0])
                y.append(class_dir)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=best_k, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)
    print("Model dilatih dan siap digunakan.")
    
    # Menjalankan kamera dengan model terbaik
    print("Starting camera...")
    run_camera(knn_clf)
