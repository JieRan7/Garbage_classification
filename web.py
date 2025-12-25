import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
import cv2

sys.path.insert(0, './yolov5-master')

MODEL_PATH = "best.pt"

print("启动系统...")

if not os.path.exists(MODEL_PATH):
    print("错误: 找不到模型文件")
    exit(1)

print("找到模型文件")

# 修复Windows路径
import platform

if platform.system() == 'Windows':
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath

# 直接使用YOLOv5的detect.py逻辑
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.augmentations import letterbox
from utils.torch_utils import select_device

# 加载模型 - 完全复制detect.py的配置
device = select_device('')
model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)  # 检查图像尺寸

# 热身模型
model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))

print(f"模型信息:")
print(f"- Stride: {stride}")
print(f"- 类别: {names}")
print(f"- 图像尺寸: {imgsz}")


def predict_image(input_image):
    try:
        # 转换为numpy数组
        if isinstance(input_image, np.ndarray):
            im0 = input_image.copy()
        else:
            im0 = np.array(input_image)

        original_shape = im0.shape

        print(f"输入图像尺寸: {original_shape}")

        # 完全复制detect.py的预处理
        im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # 填充调整
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        # 转换为tensor - 与detect.py完全一致
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0-255 to 0.0-1.0

        if len(im.shape) == 3:
            im = im[None]  # 扩展批次维度

        print(f"预处理后: {im.shape}, dtype: {im.dtype}")

        # 推理 - 使用与detect.py相同的参数
        visualize = False
        pred = model(im, augment=False, visualize=visualize)

        # NMS - 使用detect.py的默认参数
        conf_thres = 0.25  # 置信度阈值
        iou_thres = 0.45  # IOU阈值
        classes = None  # 所有类别
        agnostic_nms = False  # 类感知NMS
        max_det = 1000  # 最大检测数

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        print(f"NMS后预测数量: {len(pred)}")

        # 处理检测结果
        detection_count = 0
        for i, det in enumerate(pred):
            if len(det):
                print(f"检测到 {len(det)} 个目标")

                # 调整边界框到原图尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 按置信度排序
                det = det[det[:, 4].argsort(descending=True)]

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    class_id = int(cls)

                    # 获取类别名称
                    if class_id < len(names):
                        class_name = names[class_id]
                    else:
                        class_name = f'class_{class_id}'

                    print(f"目标 {detection_count + 1}: {class_name} {conf:.3f} [{x1},{y1},{x2},{y2}]")

                    # 绘制边界框
                    color = (0, 255, 0)
                    line_thickness = max(1, min(im0.shape[:2]) // 200)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, line_thickness)

                    # 绘制标签
                    label = f'{class_name} {conf:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness)

                    label_x2 = x1 + label_size[0][0] + 5
                    label_y1 = max(y1 - label_size[0][1] - 5, 0)
                    label_y2 = label_y1 + label_size[0][1] + 5

                    # 标签背景
                    cv2.rectangle(im0, (x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
                    # 标签文字
                    cv2.putText(im0, label, (x1 + 2, label_y1 + label_size[0][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), line_thickness)

                    detection_count += 1
            else:
                print("未检测到任何目标")

        print(f"总共绘制 {detection_count} 个检测框")
        return Image.fromarray(im0)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return input_image


# 创建界面
with gr.Blocks(theme=gr.themes.Soft(), title="垃圾识别系统") as demo:
    gr.Markdown("# 垃圾识别系统")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="上传图片",
                type="numpy",
                height=400
            )
            with gr.Row():
                conf_slider = gr.Slider(0.01, 0.99, value=0.25, label="置信度阈值")
                iou_slider = gr.Slider(0.01, 0.99, value=0.45, label="IOU阈值")
            btn = gr.Button("开始识别", variant="primary")

        with gr.Column():
            output_img = gr.Image(
                label="识别结果",
                height=400
            )

    debug_output = gr.Textbox(label="调试信息", lines=4)


    def predict_with_params(image, conf_thres, iou_thres):
        global conf_thres_global, iou_thres_global
        conf_thres_global = conf_thres
        iou_thres_global = iou_thres
        result = predict_image(image)
        return result, "处理完成，查看控制台获取详细信息"


    btn.click(fn=predict_with_params, inputs=[input_img, conf_slider, iou_slider],
              outputs=[output_img, debug_output])
    input_img.change(fn=predict_with_params, inputs=[input_img, conf_slider, iou_slider],
                     outputs=[output_img, debug_output])

print("启动Web服务...")
print("访问: http://localhost:7860")
demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)