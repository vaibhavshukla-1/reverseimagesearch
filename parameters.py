"""
This module contains the default parameters 
"""

# Path where dataset is stored (used for creating face embedding)
DATASET_PATH = "demo/"  # Path to new dataset that contains only new images

# The path where new dlib face encodings will be stored
LATEST_ENCODING_PATH = 'dataset/dlib_face_encoding_new.pkl' 

# The path where dlib face encodings are stored
DLIB_FACE_ENCODING_PATH = 'dataset/dlib_face_encoding.pkl'
# Face embedding model
FACE_EMBEDDING_MODEL = 'large'

# Number of jitters (random shifts) to use when creating the face encoding
NUMBER_OF_JITTERS = 1 
