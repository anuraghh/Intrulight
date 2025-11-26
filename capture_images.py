import cv2, os

name=input("Enter name: ").strip()
save_path=f"dataset/{name}"
os.makedirs(save_path,exist_ok=True)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

count=0
print("\n📸 Capturing 30 images... Press 'q' to quit early.\n")

while count<30:
    ret,frame=cap.read()
    if not ret:
        break

    cv2.imshow("Image Capture - Press 'q' to stop",frame)
    img_path=f"{save_path}/{count}.jpg"
    cv2.imwrite(img_path,frame)
    count+=1

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Saved {count} images in '{save_path}'")
