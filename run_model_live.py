
import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np

# Load model
with open(r"C:\Users\yarev\GestuText\model.p", 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Class label mapping (text, not emoji)
labels_map = {
    "0": "OK",
    "1": "Yes",
    "2": "No",
    "3": "Peace Out"
}

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
draw = mp.solutions.drawing_utils

# Webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            x_range = max(x_) - min(x_)
            y_range = max(y_) - min(y_)

            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - min(x_)) / x_range if x_range else 0)
                data_aux.append((lm.y - min(y_)) / y_range if y_range else 0)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_class = str(prediction[0])
            label_text = labels_map.get(predicted_class, "Unknown")

            # Display the label text (OK, Yes, No, Peace Out)
            cv.putText(frame, f'{label_text}', (10, 40), cv.FONT_HERSHEY_SIMPLEX,
                       1.5, (0, 255, 0), 4, cv.LINE_AA)

            draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv.putText(frame, "Press Q to quit", (10, 470), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv.imshow('GestuText - Live Prediction', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
