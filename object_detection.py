# -------------->  IMPORT LIBRARY
import cv2
import PIL
import urllib
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils # helper function

# -----------> 3D OBJECT DETECTION FROM IMAGES GIVEN IN URL

# Define a helper method to fetch images given a URL

def url_to_array(url):
    req = urllib.request.urlopen(url)
    arr = np.array(bytearray(req.read()), dtype=np.int8)
    arr = cv2.imdecode(arr, -1)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr

# --------------> LOAD AN IMAGE FROM A URL
mug = 'https://t4.ftcdn.net/jpg/06/19/97/81/360_F_619978183_5XXYY6Szc8paQrDBG1UNZsRtCepvWD5.jpg';
#mug = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfxPedWjxkXJc2auRUiKEWahf_7ONYV_JkFQ&s.jpg&#39;
#mug = 'https://t4.ftcdn.net/jpg/03/21/42/87/360_F_321428779_l8oQrLjoqZM5TIW22IAapbpOg0wgKufw.jpg&#39;

mug = url_to_array(mug)

# ------> LETS INSTANTIATE AN OBJECTRON INSTANCE AND PROCESS() INPUT FUNCTION

objectron = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.2,
    model_name='Cup')  # Other model names: 'Shoe', 'Cup', 'Camera', 'Laptop'

# Inference
results = objectron.process(mug)

#--------------> DISPLAY THE RESULT

if not results.detected_objects:
    print(f'No box landmarks detected.')

# Copy image so as not to draw on the original one
annotated_image = mug.copy()
for detected_object in results.detected_objects:
    # Draw landmarks
    mp_drawing.draw_landmarks(annotated_image,
                              detected_object.landmarks_2d,
                              mp_objectron.BOX_CONNECTIONS)

    # Draw axis based on rotation and translation
    mp_drawing.draw_axis(annotated_image,
                         detected_object.rotation,
                         detected_object.translation)

#-----------> PLOT THE RESULT
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(annotated_image)
ax.axis('off')
plt.show()