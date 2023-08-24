import pickle
import cv2
import face_recognition
from parameters import LATEST_ENCODING_PATH, \
                       NUMBER_OF_JITTERS, \
                       FACE_EMBEDDING_MODEL

def create_face_embeddings(image_path):
    '''
    This function creates face encodings for a single image
    '''
    #print("image path", image_path)
    #print("encod ", LATEST_ENCODING_PATH)

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(image_path)

    encoding = face_recognition.face_encodings(image, 
                                               num_jitters=NUMBER_OF_JITTERS, # Higher number of jitters increases the accuracy of the encoding
                                               model=FACE_EMBEDDING_MODEL)[0] #model='large' or 'small'
    knownEncodings.append(encoding)
    knownNames.append(image_path)  # You can use the image path as the "name"

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open(LATEST_ENCODING_PATH, "wb") as f:
        pickle.dump(data, f)
    #print("Successful")

if __name__ == '__main__':
    image_path = 'path_to_your_image.jpg'  # Provide the path to your image
    create_face_embeddings(image_path)

