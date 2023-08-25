import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import numpy as np
import base64
from dlib_face_embeddings import create_face_embeddings
from dlib_face_embeddings import merge_face_embeddings
from dlib_face_embeddings import create_face_embeddings_folder
from test import find_most_similar_images_demo
from parameters import LATEST_ENCODING_PATH, DLIB_FACE_ENCODING_PATH
from werkzeug.utils import secure_filename
import json

app = Flask(__name__, static_folder='static')
app.config["IMAGE_UPLOADS"] = "testing/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

# Assuming t1.create_embedding returns an image embedding

def get_filename_without_extension(filename):
    return os.path.splitext(filename)[0]


def allowed_image(filename: str):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1]

    if extension.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False
        
@app.route('/download_report', methods=["POST"])
def download_report():
    try:
        data = request.get_json()

        uploaded_face_name = data['uploaded']['filename']
        similar_face_name = data['similar']['filename']
        similarity_value = data['similar']['similarity']

        csv_data = f"Uploaded Face,Similar Face,Similarity Value\n{uploaded_face_name},{similar_face_name},{similarity_value}"

        response = Response(csv_data, content_type='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=report.csv'

        return response
    except Exception as e:
        return "Error generating CSV report"

            
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.files:
            image = request.files['image']
            if image.filename == '':
                return "No image selected"
            if not allowed_image(image.filename):
                return "Unsupported image format. Please use JPEG, JPG, or PNG."

            try:
                #print("Creating embedding of folder")
                create_face_embeddings_folder('testing/')
                image_filename = secure_filename(image.filename)
                image_path = os.path.join(app.config["IMAGE_UPLOADS"],image_filename)
                image.save(image_path)
                uploaded_image = cv2.imread(image_path)
                #print("image path ",image_path)

                # Call the create_embedding function from t1.py
                #print("Creating embedding of uploaded image")
                create_face_embeddings(image_path)
                
                
                # Call the find_most_similar function from t2.py
                print("fetching results")
                results = find_most_similar_images_demo(LATEST_ENCODING_PATH)
                print("result received")
                most_similar_image_path = results[0]
                similarity = results[1]
                #print("most similar ",most_similar_image_path)
                
                # Call find first image function
                result_image = cv2.imread(most_similar_image_path)

                # Encode both uploaded and similar images
                _, uploaded_img_encoded = cv2.imencode('.jpg', uploaded_image)
                _, similar_img_encoded = cv2.imencode('.jpg', result_image)
                
                uploaded_image_filename = get_filename_without_extension(os.path.basename(image_path))
                similar_image_filename = get_filename_without_extension(os.path.basename(most_similar_image_path))
                
                # ... (rest of the code)
                print("Merging embeddings")
                merge_face_embeddings(LATEST_ENCODING_PATH,DLIB_FACE_ENCODING_PATH)
                print("merge successful")
                return jsonify({
                    'uploaded': {
                        'image': base64.b64encode(uploaded_img_encoded).decode('utf-8'),
                        'filename': uploaded_image_filename
                    },
                    'similar': {
                        'image': base64.b64encode(similar_img_encoded).decode('utf-8'),
                        'path': most_similar_image_path,
                        'filename': similar_image_filename,
                        'similarity': similarity
                    }
                })
            except Exception as e:
                print(e)
                return "An error occurred during inference"

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

