import re
from keras.models import load_model
from scipy.spatial import distance
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
from script.fx import prewhiten, l2_normalize

model_path = './data/model/facenet_keras.h5'
face_cascade_path = './data/cascade/haarcascade_frontalface_default.xml'
font_path = '../data/font/Calibri Regular.ttf'
embedding_path = './data/arrays/embeddings.npz'
vars_path = './data/arrays/vars.npz'

model = load_model(model_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
loaded_embeddings = np.load(embedding_path)
embeddings, names = loaded_embeddings['a'], loaded_embeddings['b']
loaded_vars = np.load(vars_path)
slope, intercept = loaded_vars['a'], loaded_vars['b']

os.system('cls')
print("\n\n\nKeep the images in './test/' directory.")
input("Press ENTER when you're ready.")

os.chdir('./test/')
dr = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if f != 'predicted']

actual_names = []
predicted_names = []

for c in dr:
    frame = cv2.imread(c)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10)
    
    for (x_face, y_face, w_face, h_face) in faces:
        
        # Margins for Face box
        dw = 0.1 * w_face
        dh = 0.2 * h_face
        
        dist = []
        for i in range(len(embeddings)):
            face_img = cv2.resize(frame[y_face:y_face+h_face, x_face:x_face+w_face], (160, 160))
            face_img = prewhiten(face_img)
            embedding = l2_normalize(model.predict(face_img.reshape(-1, 160, 160, 3)))
            dist.append(distance.euclidean(embedding, embeddings[i].reshape(1, 128)))
        dist = np.array(dist)
        if dist.min() > 1:
            name = 'Unidentified'
        else:   
            name = names[dist.argmin()]
        print(name, dist.min())
        
        # Extract actual name from subfolder name
        actual_name = os.path.basename(os.path.dirname(c)).strip().lower()
        actual_names.append(actual_name)
        name_clean = re.sub(r'[^a-zA-Z\s]', '', name).strip().lower()
        predicted_names.append(name_clean)
        
        print(f"Actual: {actual_name}, Predicted: {name_clean}")
        
        if name != 'Unidentified':
            font_size = int(slope[dist.argmin()] * ((w_face + 2 * dw) // 3) * 2 + intercept[dist.argmin()])
        else:
            font_size = int(0.1974311 * ((w_face + 2 * dw) // 3) * 2 + 0.03397702412218706)
        
        font = ImageFont.truetype(font_path, font_size)
        size = font.getsize(name)

        cv2.rectangle(frame,
                      pt1=(x_face - int(np.floor(dw)), (y_face - int(np.floor(dh)))),
                      pt2=((x_face + w_face + int(np.ceil(dw))), (y_face + h_face + int(np.ceil(dh)))),
                      color=(0, 255, 0),
                      thickness=2) # Face Rectangle
        
        cv2.rectangle(frame,
                      pt1=(x_face - int(np.floor(dw)), y_face - int(np.floor(dh)) - size[1]),
                      pt2=(x_face + size[0], y_face - int(np.floor(dh))),
                      color=(0, 255, 0),
                      thickness=-1) # Face Name background rectangle
        
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((x_face - int(np.floor(dw)), y_face - int(np.floor(dh)) - size[1]), name, font=font, fill=(255, 0, 0))
        frame = np.array(img)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imsave('./predicted/' + os.path.basename(c), frame)

# Calculate accuracy
correct_predictions = sum([1 for actual, predicted in zip(actual_names, predicted_names) if actual == predicted])
accuracy = correct_predictions / len(actual_names) * 100
print(f"Accuracy: {accuracy:.2f}%")
print("\n\n\nOutputs can be found in './test/predicted/'")