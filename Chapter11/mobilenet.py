from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the pretrained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Load an image for classification
img_path = 'sample_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Perform classification
predictions = model.predict(x)
print('Predicted:', decode_predictions(predictions, top=3)[0])
