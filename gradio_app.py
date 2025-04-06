# Ultralytics YOLO11 Gradio Web界面
# 集成run.py和train.py功能的Web界面

# 在导入部分添加requests库
import os
import sys
import gradio as gr
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
from ultralytics import YOLO
import requests
import base64
import io
import json

# 设置默认模型路径
DEFAULT_MODEL_PATH = r'D:\yolo\yolo11\ultralytics\plantdesease\best.pt'

# 检查默认模型是否存在，如果不存在则使用yolo11n.pt
if not os.path.exists(DEFAULT_MODEL_PATH):
    DEFAULT_MODEL_PATH = 'yolo11n.pt'

# 获取当前工作目录
CWD = os.getcwd()
print(f"当前工作目录: {CWD}")

# 可用的预训练模型列表
AVAILABLE_MODELS = [
    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
    DEFAULT_MODEL_PATH
]

# 创建临时目录用于保存结果
TEMP_DIR = os.path.join(tempfile.gettempdir(), "yolo_results")
os.makedirs(TEMP_DIR, exist_ok=True)

# 推理函数
def predict(input_image, model_choice, conf_threshold):
    try:
        if input_image is None:
            return None, "请上传图像"
        
        # 加载模型
        model = YOLO(model_choice)
        
        # 执行推理
        results = model.predict(
            source=input_image,
            conf=conf_threshold,
            save=True,
            save_dir=TEMP_DIR
        )
        
        # 打印结果类型以便调试
        print(f"结果类型: {type(results)}")
        
        # 获取结果图像
        if isinstance(results, list) and len(results) > 0:
            result_image = results[0].plot()
            return Image.fromarray(result_image), "处理完成"
        elif hasattr(results, 'plot'):  # 如果results是单个结果对象
            result_image = results.plot()
            return Image.fromarray(result_image), "处理完成"
        else:
            # 确保返回值类型正确
            return None, "未检测到任何目标或结果格式不正确"
    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 打印完整错误信息到控制台
        return None, f"错误: {str(e)}"

# 训练函数
def train(model_choice, data_path, epochs, batch_size, image_size):
    try:
        # 加载模型
        model = YOLO(model_choice)
        
        # 执行训练
        model.train(
            data=data_path,
            epochs=int(epochs),
            batch=int(batch_size),
            imgsz=int(image_size)
        )
        
        return f"训练完成! 最佳模型保存在: {model.trainer.best}"
    except Exception as e:
        return f"训练错误: {str(e)}"

# 创建检测界面
def detection_interface():
    with gr.Blocks() as detection_ui:
        gr.Markdown("## YOLO11 目标检测")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="上传图像")
                model_choice = gr.Dropdown(choices=AVAILABLE_MODELS, value=DEFAULT_MODEL_PATH, label="选择模型")
                conf_threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="置信度阈值")
                detect_button = gr.Button("开始检测")
            
            with gr.Column():
                output_image = gr.Image(label="检测结果")
                output_text = gr.Textbox(label="状态")
        
        detect_button.click(
            fn=predict,
            inputs=[input_image, model_choice, conf_threshold],
            outputs=[output_image, output_text]
        )
    
    return detection_ui

# 创建训练界面
def training_interface():
    with gr.Blocks() as training_ui:
        gr.Markdown("## YOLO11 模型训练")
        
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(choices=AVAILABLE_MODELS, value="yolo11n.pt", label="选择预训练模型")
                data_path = gr.Textbox(value="data.yaml", label="数据配置文件路径")
                epochs = gr.Number(value=100, label="训练轮数")
                batch_size = gr.Number(value=16, label="批次大小")
                image_size = gr.Number(value=640, label="图像大小")
                train_button = gr.Button("开始训练")
            
            with gr.Column():
                output_text = gr.Textbox(label="训练状态", value="等待开始训练...")
        
        train_button.click(
            fn=train,
            inputs=[model_choice, data_path, epochs, batch_size, image_size],
            outputs=output_text
        )
    
    return training_ui

# 创建主界面
# 添加调用视觉大模型的函数
# 修改analyze_plant_disease函数以支持流式输出
def analyze_plant_disease(image):
    try:
        if image is None:
            return "请上传图像"
        
        # 将图片转换为base64编码
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_url = f"data:image/jpeg;base64,{img_str}"
        
        url = "https://api.siliconflow.cn/v1/chat/completions"
        
        payload = {
            "model": "Qwen/Qwen2.5-VL-32B-Instruct",
            "stream": True,  # 启用流式输出
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": [],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image_url": {
                                "detail": "auto",
                                "url": img_url
                            },
                            "type": "image_url"
                        },
                        {
                            "text": "这是什么病虫害？同时给出预防方法",
                            "type": "text"
                        }
                    ]
                }
            ]
        }
        
        headers = {
            "Authorization": "Bearer sk-jivwbgqsesocbzkggntyzjwlkvlyhuiaphesburlvyswzsfc",
            "Content-Type": "application/json"
        }
        
        # 使用yield实现流式输出
        response = requests.request("POST", url, json=payload, headers=headers, stream=True)
        
        if response.status_code == 200:
            collected_chunks = []
            for chunk in response.iter_lines():
                if chunk:
                    chunk_str = chunk.decode('utf-8')
                    if chunk_str.startswith('data: '):
                        chunk_data = chunk_str[6:]  # 去掉 'data: ' 前缀
                        if chunk_data != '[DONE]':
                            try:
                                json_data = json.loads(chunk_data)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        collected_chunks.append(content)
                                        yield ''.join(collected_chunks)
                            except json.JSONDecodeError:
                                pass
            
            # 如果没有收到任何内容，返回错误信息
            if not collected_chunks:
                yield "未能获取到有效的分析结果"
            else:
                yield ''.join(collected_chunks)
        else:
            yield f"API请求失败: {response.status_code} - {response.text}"
    
    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        yield f"分析过程中出错: {str(e)}"

# 修改病虫害分析界面以支持流式输出和Markdown预览
def disease_analysis_interface():
    with gr.Blocks() as analysis_ui:
        gr.Markdown("## 植物病虫害智能分析")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="上传植物图像")
                analyze_button = gr.Button("分析病虫害")
            
            with gr.Column():
                # 使用Markdown组件替代Textbox以支持Markdown渲染
                output_text = gr.Markdown(label="分析结果")
        
        analyze_button.click(
            fn=analyze_plant_disease,
            inputs=input_image,
            outputs=output_text
        )
    
    return analysis_ui

# 创建主界面
with gr.Blocks(title="YOLO11 Web界面") as app:
    gr.Markdown("# YOLO11 目标检测与训练界面")
    gr.Markdown("使用YOLO11进行目标检测和模型训练的Web界面")
    
    with gr.Tabs():
        with gr.TabItem("目标检测"):
            detection_interface()
        
        with gr.TabItem("模型训练"):
            training_interface()
            
        with gr.TabItem("病虫害智能分析"):
            disease_analysis_interface()

# 修改主界面启动代码，移除不兼容的reload参数
if __name__ == "__main__":
    # 使用简单的界面启动方式
    app.launch(share=True)

# 删除以下重复的代码块，因为它们会导致冲突
# def create_interface():
#     with gr.Blocks() as demo:
#         gr.Markdown("# YOLO11 目标检测演示")
#         
#         with gr.Row():
#             with gr.Column():
#                 input_image = gr.Image(type="pil", label="上传图像")
#                
#                 with gr.Row():
#                     model_dropdown = gr.Dropdown(
#                         choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
#                         value="yolov8n.pt",
#                         label="选择模型"
#                     )
#                     conf_slider = gr.Slider(
#                         minimum=0.1, 
#                         maximum=1.0, 
#                         value=0.25, 
#                         step=0.05, 
#                         label="置信度阈值"
#                     )
#                
#                 submit_btn = gr.Button("开始检测")
#             
#             with gr.Column():
#                 output_image = gr.Image(type="pil", label="检测结果")
#                 output_text = gr.Textbox(label="状态")
#         
#         submit_btn.click(
#             fn=predict,
#             inputs=[input_image, model_dropdown, conf_slider],
#             outputs=[output_image, output_text]
#         )
#     
#     return demo

# 启动应用
# if __name__ == "__main__":
#     demo = create_interface()
#     demo.launch()