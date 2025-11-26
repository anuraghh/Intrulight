import os, cv2, numpy as np, pickle
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 100
dataset_path = "dataset"

# Load model and encoder
model = load_model("face_recognition_model.h5")
with open("label_encoder.pkl","rb") as f:
    encoder = pickle.load(f)

# Load dataset again for testing
X, y = [], []
for label in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, label)
    if not os.path.isdir(person_dir): continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

X = np.array(X, dtype="float32") / 255.0
y_encoded = encoder.transform(y)
y_onehot = to_categorical(y_encoded)

# Split data for testing
_, X_test, _, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")
