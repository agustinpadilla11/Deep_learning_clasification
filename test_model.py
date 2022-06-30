from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

names = ['cat', 'dog']

modelt = load_model('models/perros-gatos.h5')

imaget_path = "imagenprueba.jpg"
imaget=cv2.resize(cv2.imread(imaget_path), (150, 150), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)

pred = modelt.predict(xt)

print(names[np.argmax(pred)])
plt.imshow(cv2.cvtColor(np.asarray(imaget),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()