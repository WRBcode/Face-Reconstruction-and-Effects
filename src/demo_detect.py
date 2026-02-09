# src/demo_detect.py
import cv2
import sys

def detect_face(image_path):
    # 加载 OpenCV 自带的人脸分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 画框
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # 保存结果
    cv2.imwrite("output_result.jpg", img)
    print(f"Success! Detected {len(faces)} faces. Saved to output_result.jpg")

if __name__ == "__main__":
    # 这里的路径你可以改成你自己的照片路径
    # 比如: python src/demo_detect.py /data/ruibo/Face-Reconstruction-and-Effects/src/mask.png
    if len(sys.argv) > 1:
        detect_face(sys.argv[1])
    else:
        print("Usage: python src/demo_detect.py <path_to_image>")