# import os
# import cv2

# face_cascade_path = '../data/cascade/haarcascade_frontalface_default.xml'
# images_path = '../data/images/'
# output_path = '../data/faces/'

# def detect_faces():
#     face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
#     if face_cascade.empty():
#         print(f'Error loading cascade file: {face_cascade_path}')
#         input()
#         quit()
    
#     if not os.path.exists(images_path):
#         print(f'\n\nDirectory {images_path} does not exist')
#         input()
#         quit()
    
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
    
#     for root, dirs, files in os.walk(images_path):
#         for file in files:
#             if file.endswith(('.jpg', '.jpeg', '.png')):
#                 img_path = os.path.join(root, file)
#                 img = cv2.imread(img_path)
                
#                 if img is None:
#                     print(f'Error reading image: {img_path}')
#                     continue
                
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                
#                 for (x, y, w, h) in faces:
#                     image = img[y:y+h, x:x+w]
#                     relative_path = os.path.relpath(root, images_path)
#                     output_dir = os.path.join(output_path, relative_path)
                    
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)
                    
#                     output_name = "{}_{}x{}_{}x{}.jpg".format(os.path.splitext(file)[0], x, y, w, h)
#                     output_file_path = os.path.join(output_dir, output_name)
#                     cv2.imwrite(output_file_path, image)
#                     print('Done Detecting: ', output_file_path)

import os
import cv2

face_cascade_path='../data/cascade/haarcascade_frontalface_default.xml'
images_path='../data/images/'

def detect_faces():
    face_cascade=cv2.CascadeClassifier(face_cascade_path)
    os.chdir(images_path)
    
    if len(os.listdir())==0:
        print('\n\nNo Images Found')
        input()
        quit()
    for i in os.listdir():
        name=i
        img=cv2.imread(i)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
    
        for (x,y,w,h) in faces:
            image=img[y:y+h,x:x+w]
            output_name="{}.faces{}x{}_{}x{}.jpg".format(name,x,y,w,h)
            cv2.imwrite('../faces/'+output_name,image)
            print('Done Detecting: ',output_name)
    os.chdir('../../script')