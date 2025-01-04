import gradio as gr
import cv2
import face_recognition
import numpy as np
import pickle
from PIL import Image, ImageDraw

# Muat model KNN yang telah dilatih
def load_model(model_path="trained_knn_model.clf"):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Fungsi prediksi wajah dengan model KNN
def predict_face(image):
    knn_clf = load_model()  # Muat model KNN
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  # Convert to RGB for face_recognition
    face_locations = face_recognition.face_locations(rgb_image)
    
    if not face_locations:
        return image, "No face detected"
    
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [dist[0] <= 0.6 for dist in closest_distances[0]]
    
    # Predict classes and remove classifications that aren't within the threshold
    predictions = knn_clf.predict(face_encodings)
    results = []
    recognized_names = []  # List to store recognized names
    
    for pred, loc, rec in zip(predictions, face_locations, are_matches):
        if rec:
            results.append((pred, loc))
            recognized_names.append(pred)
        else:
            results.append(("Unknown", loc))
            recognized_names.append("Unknown")
    
    # Draw results on the image
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in results:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=2)
        draw.text((left + 6, bottom - 6), name, fill=(255, 255, 255, 255))
    
    # Convert back to BGR to maintain original color for OpenCV display
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return final_image, ", ".join(recognized_names)

# Define Gradio interface
iface = gr.Interface(
    fn=predict_face, 
    inputs="image", 
    outputs=["image", "text"],  # Multiple outputs: annotated image and names
    live=True
)

# Jalankan aplikasi Gradio
if __name__ == "__main__":
    iface.launch()

# import gradio as gr
# import cv2
# import face_recognition
# import numpy as np
# import pickle
# from PIL import Image, ImageDraw

# # Muat model KNN yang telah dilatih
# def load_model(model_path="trained_knn_model.clf"):

#     with open(model_path, 'rb') as f:
#         return pickle.load(f)

# # Fungsi prediksi wajah dengan model KNN
# def predict_face(image, knn_clf):
#     rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  # Convert to RGB for face_recognition
#     face_locations = face_recognition.face_locations(rgb_image)
#     if not face_locations:
#         return "No face detected"
    
#     face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
#     closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
#     are_matches = [dist[0] <= 0.6 for dist in closest_distances[0]]
    
#     # Predict classes and remove classifications that aren't within the threshold
#     predictions = knn_clf.predict(face_encodings)
#     results = []
#     for pred, loc, rec in zip(predictions, face_locations, are_matches):
#         if rec:
#             results.append((pred, loc))
#         else:
#             results.append(("Unknown", loc))
    
#     # Draw results on the image
#     pil_image = Image.fromarray(rgb_image)
#     draw = ImageDraw.Draw(pil_image)
#     for name, (top, right, bottom, left) in results:
#         draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
#         draw.text((left + 6, bottom - 6), name, fill=(255, 255, 255, 255))
    
#     return pil_image

# # Load the model once
# knn_clf = load_model()

# # Define Gradio interface
# iface = gr.Interface(fn=lambda img: predict_face(img, knn_clf), inputs="image", outputs="image", live=True)

# # Jalankan aplikasi Gradio
# if __name__ == "__main__":
#     iface.launch()