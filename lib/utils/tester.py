# 文件路径: lib/utils/tester.py

from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import os
from .decode import ctdet_decode
from .post_process import ctdet_post_process

def process_frame(model, image, opt, device):
    """
    对单个图像帧进行模型推理。
    接收一个图像张量，返回原始的检测结果张量。
    """
    with torch.no_grad():
        image = image.to(device)
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output.get('reg', None)
        dets = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    return dets

def post_process_frame(dets, meta, num_classes):
    """
    对单帧的检测结果进行后处理。
    将坐标从输出热图尺度转换到调整后（resized）的输入图像尺度。
    """
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    
    # ctdet_post_process 使用 meta 字典中的 'c' (中心) 和 's' (尺度)
    # 来进行从输出热图到输入图像的逆仿射变换。
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    
    # 将结果整理成按类别组织的字典
    for j in range(1, num_classes + 1):
        if len(dets[0][j]) > 0:
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        else:
            dets[0][j] = np.empty([0, 5], dtype=np.float32)
            
    return dets[0]

def save_yolo_format(predictions, resized_shape, original_img_shape, save_path, coco_id_to_yolo_id_map, conf_thresh):
    """
    将处理后的检测结果保存为YOLO格式的 .txt 文件。
    - 接收在 resized 图像尺度下的坐标。
    - 将坐标缩放到原始图像尺度。
    - 根据置信度阈值进行筛选。
    - 将结果归一化并保存。
    """
    resized_h, resized_w = resized_shape
    original_h, original_w = original_img_shape
    
    # 计算从 resized 尺寸到 original 尺寸的缩放比例
    scale_w = original_w / resized_w if resized_w > 0 else 1
    scale_h = original_h / resized_h if resized_h > 0 else 1

    with open(save_path, 'w') as f:
        # predictions 的 key 是 COCO 类别 ID (从1开始)
        for coco_cls_id in predictions:
            yolo_cls_id = coco_id_to_yolo_id_map.get(coco_cls_id)
            if yolo_cls_id is None:
                continue

            for bbox in predictions[coco_cls_id]:
                score = bbox[4]
                
                # 在这里进行置信度筛选
                if score >= conf_thresh:
                    # bbox[:4] 的坐标是基于 resized_shape (例如 640x512)
                    x1_resized, y1_resized, x2_resized, y2_resized = bbox[:4]

                    # 将坐标从 resized 空间线性缩放到 original 图像空间
                    x1_orig = x1_resized * scale_w
                    y1_orig = y1_resized * scale_h
                    x2_orig = x2_resized * scale_w
                    y2_orig = y2_resized * scale_h
                    
                    # 使用原始图像尺寸进行裁剪，确保坐标在有效范围内
                    x1, x2 = np.clip([x1_orig, x2_orig], 0, original_w - 1)
                    y1, y2 = np.clip([y1_orig, y2_orig], 0, original_h - 1)

                    box_w, box_h = x2 - x1, y2 - y1

                    # 确保宽度和高度有效
                    if box_w <= 0 or box_h <= 0:
                        continue

                    center_x = x1 + box_w / 2
                    center_y = y1 + box_h / 2
                    
                    # 使用原始图像尺寸进行归一化
                    center_x_norm = center_x / original_w
                    w_norm = box_w / original_w
                    center_y_norm = center_y / original_h
                    h_norm = box_h / original_h

                    f.write(
                        f"{yolo_cls_id} {center_x_norm:.6f} {center_y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}\n")