# 开始训练模型
from ultralytics import YOLO
import multiprocessing
import os

# 加载预训练模型
def main():
    a1=YOLO('yolo11n.pt')
    
    # 打印当前工作目录，帮助调试
    print(f"当前工作目录: {os.getcwd()}")
    
    a1.train(
        data='data.yaml', # 数据集配置文件路径
        epochs=500, # 训练次数
        imgsz=640, # 输入图片尺寸
        batch=32, # 训练批次
        device=0,
        amp=False, # 禁用自动混合精度
        val=False, # 暂时禁用验证，直到修复数据集路径
    )
    print('训练完毕')

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 添加多进程支持
    main()