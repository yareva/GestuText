import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Running train_model.py...")

# Load data
data_dict = pickle.load(open(r"C:\Users\yarev\GestuText\data.pickle", 'rb'))

data = np.asarray(data_dict['data'])
label = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, label, test_size=0.2, shuffle=True, stratify=label
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print("Data shape:", data.shape)
print("Labels shape:", label.shape)
print("Unique labels:", np.unique(label))
print("Sample data:", data[:5])
print("Sample labels:", label[:5])
print('{}% of samples were classified correctly!'.format(score * 100))

# Save model
with open(r"C:\Users\yarev\GestuText\model.p", 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved to model.p")
