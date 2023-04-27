from flask import Flask
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
  # Load the model
  model = load_model('keras_model.h5', compile=False)


# Load the labels



#Path of the Image
  img_path='absolute/path/to/testk1.jpg'

  image=cv2.imread(img_path)

# Resize the raw image into (224-height,224-width) pixels
  image = cv2.resize(image, (224, 224),             
  interpolation=cv2.INTER_AREA)

# Make the image a numpy array and reshape it to the models input shape.
  image = np.asarray(image, dtype=np.float32).reshape(1,  224, 224, 3)

# Normalize the image array
  image = (image / 127.5) - 1

# Predicts the model
  prediction = model.predict(image)
  #index = np.argmax(prediction)
  

  print("prediction",prediction)

  confidence_score = prediction[0][index]
  print("Class index",index)
 
  print("Accuracy",confidence_score)

  
  return class_name
if __name__ == "__main__":
  app.run(host='0.0.0.0',debug=True)