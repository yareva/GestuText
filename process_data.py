
import os
import pickle


import mediapipe as mp
import cv2


# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
   static_image_mode=True,
   min_detection_confidence=0.3,
   model_complexity=1  # Ensures a balance between performance and accuracy
)


# Dataset directory
DATA_DIR = r"C:\Users\yarev\GestuTextData"


# Clean up .DS_Store files
for root, dirs, files in os.walk(DATA_DIR):
   if '.DS_Store' in files:
       os.remove(os.path.join(root, '.DS_Store'))


# Preprocessing function to ensure a square input image
def preprocess_image(img):
   h, w, _ = img.shape
   size = max(h, w)  # Get the size of the largest dimension
   square_img = cv2.copyMakeBorder(
       img,
       (size - h) // 2, (size - h + 1) // 2,  # Top, Bottom padding
       (size - w) // 2, (size - w + 1) // 2,  # Left, Right padding
       cv2.BORDER_CONSTANT,
       value=[0, 0, 0]  # Black padding
   )
   return square_img


# Initialize data and labels
data = []
labels = []


# Process images
for dir_ in os.listdir(DATA_DIR):
   dir_path = os.path.join(DATA_DIR, dir_)
   if not os.path.isdir(dir_path):
       continue  # Skip non-directory files


   for img_path in os.listdir(dir_path):
       img = cv2.imread(os.path.join(dir_path, img_path))
       if img is None:
           print(f"Error loading image: {os.path.join(dir_path, img_path)}")
           continue


       img = preprocess_image(img)  # Ensure the image is square
       img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


       # Process image with Mediapipe Hands
       results = hands.process(img_rgb)
       if results.multi_hand_landmarks:
           for hand_landmarks in results.multi_hand_landmarks:
               data_aux = []
               x_ = [lm.x for lm in hand_landmarks.landmark]
               y_ = [lm.y for lm in hand_landmarks.landmark]


               if x_ and y_:
                   x_range = max(x_) - min(x_)
                   y_range = max(y_) - min(y_)


                   for lm in hand_landmarks.landmark:
                       # Normalize coordinates to fit in range [0, 1]
                       data_aux.append((lm.x - min(x_)) / x_range if x_range else 0)
                       data_aux.append((lm.y - min(y_)) / y_range if y_range else 0)


                   data.append(data_aux)
                   labels.append(dir_)


# Save the dataset
with open('data.pickle', 'wb') as f:
   pickle.dump({'data': data, 'labels': labels}, f)


print("Dataset creation complete. Data saved in 'data.pickle'.")


