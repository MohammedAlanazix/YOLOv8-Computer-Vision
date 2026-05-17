import cv2
from ultralytics import YOLO

# 1. تحميل النموذج المدرب (YOLOv8)
model = YOLO('yolov8n.pt') 

# 2. تحديد مسار الفيديو أو الصورة المُراد رصدها
source_path = 'cars_video.mp4' 

# 3. تشغيل الرصد وتوليد المربعات المحيطة بالأجسام
results = model.predict(source=source_path, save=True, conf=0.5)

print("تمت عملية الرصد وتحديد الأجسام بنجاح وبدقة عالية!")
