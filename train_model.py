import os, cv2, torch, joblib
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

DATASET_DIR=Path("dataset")
MODEL_DIR=Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("🧠 Starting training process...")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector=MTCNN(keep_all=False,device=device)
    facenet=InceptionResnetV1(pretrained='vggface2').eval().to(device)

    embeddings,labels=[],[]

    for person in os.listdir(DATASET_DIR):
        person_path=DATASET_DIR/person
        if not os.path.isdir(person_path): continue
        print(f"🔍 Processing: {person}")

        for img_file in os.listdir(person_path):
            img_path=str(person_path/img_file)
            img=cv2.imread(img_path)
            if img is None: continue

            boxes,_=detector.detect(img)
            if boxes is None or len(boxes)==0: continue

            x1,y1,x2,y2=[int(b) for b in boxes[0]]
            face=img[y1:y2,x1:x2]
            if face.size==0: continue

            face=cv2.resize(face,(160,160))
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=Image.fromarray(face)

            face_tensor=torch.tensor(np.array(face)/255.).permute(2,0,1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                emb=facenet(face_tensor).cpu().numpy()[0]

            embeddings.append(emb)
            labels.append(person)

    if not embeddings:
        print("❌ No faces detected in dataset.")
        return

    print(f"✅ Collected {len(embeddings)} embeddings for {len(set(labels))} people.")

    encoder=LabelEncoder()
    y=encoder.fit_transform(labels)

    svm=SVC(kernel='linear',probability=True)
    svm.fit(embeddings,y)

    joblib.dump(svm,MODEL_DIR/"svm_model.pkl")
    joblib.dump(encoder,MODEL_DIR/"label_encoder.pkl")

    print("🎉 Training completed and models saved!")

if __name__=="__main__":
    main()
