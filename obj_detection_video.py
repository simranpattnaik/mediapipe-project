# ----------> IMPORT LIBRARIES
import cv2
import mediapipe as mp

# ----------> MEDIAPIPE SETUP
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# ----------> VIDEO SOURCE
# For webcam, use: cv2.VideoCapture(0)
cap = cv2.VideoCapture(
    r"laptop_kitchen.mp44"
)

# ----------> OBJECTRON MODEL (MATCH THE OBJECT!)
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=2,
    min_detection_confidence=0.25,
    min_tracking_confidence=0.6,
    model_name='Laptop'   # IMPORTANT CHANGE
)

# ----------> CREATE RESIZABLE WINDOW
cv2.namedWindow("MediaPipe Objectron", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MediaPipe Objectron", 960, 540)

frame_count = 0

# ----------> PROCESS VIDEO
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Video ended or cannot read frame.")
        break

    frame_count += 1

    # Optional: skip frames for stability
    if frame_count % 2 != 0:
        continue

    # Improve performance
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inference
    results = objectron.process(image_rgb)

    # Draw results
    image.flags.writeable = True

    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Draw 2D box landmarks
            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )

            # Draw 3D axis
            mp_drawing.draw_axis(
                image,
                detected_object.rotation,
                detected_object.translation
            )

    # ----------> RESIZE FRAME TO FIT SCREEN
    image = cv2.resize(image, (960, 540))

    # ----------> SHOW OUTPUT
    cv2.imshow("MediaPipe Objectron", image)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ----------> CLEANUP
cap.release()
cv2.destroyAllWindows()