import pickle

with open ('dlib_face_encoding_new.pkl','rb') as f:
	data=pickle.load(f)
print(data)
