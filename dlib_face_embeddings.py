import pickle
import cv2
import face_recognition
from parameters import LATEST_ENCODING_PATH, \
                       NUMBER_OF_JITTERS, \
                       FACE_EMBEDDING_MODEL, \
                       DLIB_FACE_ENCODING_PATH
import os

def merge_face_embeddings(LATEST_ENCODING_PATH,DLIB_FACE_ENCODING_PATH):

    with open(LATEST_ENCODING_PATH, "rb") as new:
        d1 = pickle.load(new)

    with open(DLIB_FACE_ENCODING_PATH, "rb") as old:
        d2 = pickle.load(old)

    merged = {}
    for key in d1:
        if key in d2:
            continue
        else:
            merged[key] = d1[key]

    # Add remaining embeddings from d2 (if any)
    for key in d2:
        if key not in d1:
            merged[key] = d2[key]

    # Save the merged embeddings to the old embeddings file
    with open(DLIB_FACE_ENCODING_PATH, "wb") as merged_embeddings:
        pickle.dump(merged, merged_embeddings)

    # Delete the new embeddings file
    if os.path.exists(LATEST_ENCODING_PATH):
        os.remove(LATEST_ENCODING_PATH)


def allowed_image(filename: str):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1]

    if extension.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def create_face_embeddings_folder(directory_path):
    '''
    This function creates face encodings for images in a directory
    '''
    # Initialize the list of known encodings and known names
   
    knownEncodings = []
    knownNames = []
    
    # List all files in the directory
    image_files = [file for file in os.listdir(directory_path) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Load the input image and convert it from BGR (OpenCV ordering)
        # to RGB (face_recognition ordering)
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate face encoding for the image
        encoding = face_recognition.face_encodings(rgb_image,
                                                   num_jitters=NUMBER_OF_JITTERS,
                                                   model=FACE_EMBEDDING_MODEL)
        
        if len(encoding) > 0:
            knownEncodings.append(encoding[0])  # We'll take the first face if there are multiple
            knownNames.append(image_file)

    # Dump the facial encodings + names to disk
    #print("[INFO] Serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(DLIB_FACE_ENCODING_PATH, "wb") as f:
        pickle.dump(data, f)
    #print("Successful for folder")
        

def create_face_embeddings(image_path):
    '''
    This function creates face encodings for a single image
    '''
    #print("image path", image_path)
    #print("encod ", LATEST_ENCODING_PATH)

    # initialize the list of known encodings and known names
    newEncodings = []
    newNames = []

    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(image_path)

    encoding = face_recognition.face_encodings(image, 
                                               num_jitters=NUMBER_OF_JITTERS, # Higher number of jitters increases the accuracy of the encoding
                                               model=FACE_EMBEDDING_MODEL)[0] #model='large' or 'small'
    image_path=os.path.basename(image_path)             
    newEncodings.append(encoding)
    #print(image_path)
    newNames.append(image_path)  # You can use the image path as the "name"
    data = {"encodings": newEncodings, "names": newNames}
    with open(LATEST_ENCODING_PATH, "wb") as f:
        pickle.dump(data, f)
    #print("Successful for uploaded image")

if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'  # Provide the path to your image
    create_face_embeddings(image_path)


