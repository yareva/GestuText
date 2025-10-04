# GestuText ✋

**GestuText** is a hand gesture recognition system using **MediaPipe**, **OpenCV**, and **scikit-learn**. It detects hand gestures from a camera in real-time and classifies them using a trained Random Forest model.

## 🤖 Supported Gestures

| Gesture   | Class | Label       |
|----------|--------|-------------|
| 👌       | 0      | OK          |
| 👍       | 1      | Yes         |
| 👎       | 2      | No          |
| ✌️       | 3      | Peace Out   |

## 📦 Project Structure

- `collect_data.py` – Captures webcam images of each gesture
- `process_data.py` – Extracts hand landmarks using MediaPipe and saves data to `data.pickle`
- `train_model.py` – Trains a Random Forest classifier and saves `model.p`
- `run_model_live.py` – Uses webcam to detect gestures in real-time and display predictions

## 🚀 How to Run

1. **Collect Data**  
   Run `collect_data.py` and follow on-screen instructions to capture 200 images per gesture.

2. **Process Data**  
   ```bash
   python process_data.py
