from ultralytics import YOLO
# 加载模型
file=r'D:\yolo\yolo11\ultralytics\models\images\train\Aug_Ferisia pseudococcus (Signoret)_158.jpg'
modles=r'D:\yolo\yolo11\ultralytics\plantdesease\best.pt'
a1=YOLO(modles)
a1(file,show=True,save=True)
