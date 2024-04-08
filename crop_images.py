import cv2
import numpy as np
import os
from PIL import Image

def setup_directories(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

def load_image_with_fallback(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise IOError("Image loading failed")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img
    except Exception as open_cv_error:
        try:
            pil_img = Image.open(image_path).convert("RGB")
            cv_img = np.array(pil_img)[..., ::-1] 
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), cv_img
        except Exception as pil_error:
            raise Exception(f"OpenCV Error: {open_cv_error}, PIL Error: {pil_error}")

def detect_and_crop_faces(source_dir, destination_dir, cascade_path):
    face_detector = cv2.CascadeClassifier(cascade_path)
    for img_file in filter(lambda f: f.lower().endswith(('.png', '.jpg')), os.listdir(source_dir)):
        img_path = os.path.join(source_dir, img_file)
        try:
            gray_img, color_img = load_image_with_fallback(img_path)
            faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for i, (x, y, w, h) in enumerate(faces):
                face_img = color_img[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(destination_dir, f"face_{i}_{img_file}"), face_img)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

def main():
    input_directory = "test" #"train"
    output_directory = "test-cropped" #"train-cropped"
    haar_cascade_filepath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    setup_directories(input_directory, output_directory)
    detect_and_crop_faces(input_directory, output_directory, haar_cascade_filepath)
    print("Cropping process done")

if __name__ == "__main__":
    main()
