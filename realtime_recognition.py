import cv2, torch, joblib, os
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path

MODEL_DIR=Path("models")
SVM_PATH=MODEL_DIR/"svm_model.pkl"
ENC_PATH=MODEL_DIR/"label_encoder.pkl"

if not (SVM_PATH.exists() and ENC_PATH.exists()):
    print("❌ Models not found. Run train_model.py first.")
    exit()

svm=joblib.load(SVM_PATH)
encoder=joblib.load(ENC_PATH)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn=MTCNN(keep_all=True,device=device)
facenet=InceptionResnetV1(pretrained='vggface2').eval().to(device)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

print("\n📷 Real-time Recognition Started (press 'x' to exit)\n")

while True:
    ret,frame=cap.read()
    if not ret: break

    boxes,_=mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1,y1,x2,y2=[int(b) for b in box]
            face=frame[y1:y2,x1:x2]
            if face.size==0: continue

            try:
                face=cv2.resize(face,(160,160))
            except: continue

            face_tensor=torch.tensor(face/255.).permute(2,0,1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                emb=facenet(face_tensor).cpu().numpy()[0]

            probs=svm.predict_proba([emb])[0]
            conf=np.max(probs)
            if conf<0.6:
                name="Unknown"
            else:
                pred=svm.predict([emb])[0]
                name=encoder.inverse_transform([pred])[0]

            color=(0,255,0) if name!="Unknown" else (0,0,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{name} ({conf*100:.1f}%)",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    cv2.imshow("Real-Time Face Recognition",frame)
    if cv2.waitKey(1)&0xFF==ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Session ended.")
