import pickle
import os
import cv2
import numpy as np
import face_recognition
from parameters import LATEST_ENCODING_PATH, FACE_EMBEDDING_MODEL, DLIB_FACE_ENCODING_PATH

def find_most_similar_images_demo(LATEST_ENCODING_PATH,num_similar_images=2):
    with open(DLIB_FACE_ENCODING_PATH, "rb") as f:
        #print("openning dlib")
        data = pickle.load(f)
        knownEncodings = data["encodings"]
        #print(len(knownEncodings))
        knownNames = data["names"]
        #print(knownNames)

    with open(LATEST_ENCODING_PATH, "rb") as d:
        #print("opening latest")
        new_data = pickle.load(d)
        demo_encodings = new_data["encodings"]
        demoNames = new_data["names"]
    results = []

    for demo_encoding, demoName in zip(demo_encodings, demoNames):
        similar_images = []
        similarities = [] 
        for knownEncoding, knownName in zip(knownEncodings, knownNames):
            similarity = face_recognition.face_distance(np.array([knownEncoding]), np.array(demo_encoding))[0]
            similarities.append(similarity)
            similar_images.append((knownName, similarity))
        similar_images.sort(key=lambda x: x[1])  # Sort by similarity
        most_similar_images = similar_images[:num_similar_images]
        similar_image_folder = os.path.join("testing/")

        for similar_image_name, similarity in most_similar_images:
            dN = os.path.basename(demoName)
            idx = dN.index('.')
            dN=dN[:idx]
            sin = os.path.basename(similar_image_name)
            idx = sin.index('.')
            sin=sin[:idx]
            if sin!= dN:
                most_similar_image_path = os.path.join(similar_image_folder, similar_image_name)
                results.append(most_similar_image_path)
                results.append(similarity)
                return results
if __name__ == '__main__':
    print(find_most_similar_images_demo(LATEST_ENCODING_PATH))

