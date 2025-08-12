from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import json
from progress.bar import Bar
import matplotlib.pyplot as plt
import matplotlib
import math
import time
from functools import partial
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

matplotlib.use('Agg')

from lib.utils.opts import opts
from lib.models.stNet import get_det_net
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from evaluator import DetectionEvaluator
import cv2
from lib.utils.image import get_affine_transform

# --- 模型加载函数 (最终修复版) ---
def load_parallel_models(model_path, model_count, opt, dataset_num_classes):
    """
    加载指定数量的模型用于并行处理，能处理多种权重文件格式，
    特别是带有 'modelX.' 前缀的扁平化多模型文件。
    """
    print(f"==> Loading weights from: {model_path}")
    print(f"==> Will create {model_count} model instances for parallel processing...")
    
    models = []
    checkpoint = torch.load(model_path, map_location=opt.device)
    
    # 修正检测逻辑：检查是否有任何键以 'model0.' 开头来判断是否为多模型文件
    is_multi_model_checkpoint = isinstance(checkpoint, dict) and any(key.startswith('model0.') for key in checkpoint.keys())

    if is_multi_model_checkpoint:
        print("==> Detected a multi-model checkpoint with prefixed keys.")
        for i in range(model_count):
            model = get_det_net({'hm': dataset_num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
            
            # 为当前模型实例提取权重
            model_state_dict = {}
            model_prefix = f'model{i}.'
            
            for key, value in checkpoint.items():
                if key.startswith(model_prefix):
                    # 剥离前缀，得到原始的层名称
                    new_key = key[len(model_prefix):]
                    model_state_dict[new_key] = value
            
            # 如果找不到对应模型的权重（例如请求的model_count > 文件中的模型数），则回退使用 model0 的权重
            if not model_state_dict:
                print(f"⚠️  Warning: No weights found for model{i}. Falling back to model0 weights.")
                model0_prefix = 'model0.'
                for key, value in checkpoint.items():
                    if key.startswith(model0_prefix):
                        new_key = key[len(model0_prefix):]
                        model_state_dict[new_key] = value

            if model_state_dict:
                model.load_state_dict(model_state_dict)
                # print(f"✅ Extracted and loaded weights for model instance {i}.")
            else:
                # 这种情况理论上不会发生，因为有回退机制
                print(f"❌ Error: Could not load any weights for model {i}. It will have random weights.")
            
            model = model.to(opt.device)
            model.eval()
            models.append(model)
        print(f"✅ Successfully created and loaded weights for all {model_count} model instances.")

    else:
        # 处理单模型文件的逻辑保持不变
        print("==> Detected a single-model weight file. All instances will share these weights.")
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            base_model_state = checkpoint['state_dict']
            print("    -> Found 'state_dict' key in the file.")
        else:
            base_model_state = checkpoint
            print("    -> Assuming the file is a direct state_dict.")

        for i in range(model_count):
            model = get_det_net({'hm': dataset_num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
            model.load_state_dict(base_model_state)
            model = model.to(opt.device)
            model.eval()
            models.append(model)
        print(f"✅ Loaded shared weights for all {model_count} model instances.")

    return models


# --- 并行推理工作函数 ---
def inference_worker(args):
    """
    使用指定模型处理分配给它的数据批次 (在线程中运行)。
    """
    model_id, model, data_batch, opt, dataset_info, predictions_buffer, lock = args
    
    local_buffer = defaultdict(list)
    
    for item in data_batch:
        seq_inp, meta, frame_path = item['seq_inp'], item['meta'], item['frame_path']
        
        image = torch.from_numpy(np.array([seq_inp])).to(opt.device)
        
        # 模型推理
        dets_raw = process(model, image, opt)
        dets = post_process(dets_raw, meta, dataset_info['num_classes'])
        
        # 处理文件路径
        path_parts = frame_path.replace('\\', '/').split('/')
        video_name = path_parts[-2] if len(path_parts) > 1 else 'video_root'
        frame_name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]

        # 生成YOLO格式的预测内容
        pred_content = save_predictions_as_yolo(
            predictions=dets,
            resized_shape=(dataset_info['default_resolution'][1], dataset_info['default_resolution'][0]),
            original_img_shape=(meta['original_height'], meta['original_width']),
            save_path=None,  # 不直接保存，返回字符串
            coco_id_to_yolo_id_map=dataset_info['coco_id_to_yolo_id']
        )
        
        local_buffer[video_name].append((frame_name_no_ext, pred_content))

    # 使用锁将本地缓存安全地合并到全局缓存中
    with lock:
        for video_name, frame_data in local_buffer.items():
            predictions_buffer[video_name].extend(frame_data)
    
    return len(data_batch)

# --- 辅助函数 ---
def process(model, image, opt):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output.get('reg', None)
        dets = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    return dets

def post_process(dets, meta, num_classes):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        if len(dets[0][j]) > 0:
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        else:
            dets[0][j] = np.empty([0, 5], dtype=np.float32)
    return dets[0]

def save_predictions_as_yolo(predictions, resized_shape, original_img_shape, save_path, coco_id_to_yolo_id_map):
    resized_h, resized_w = resized_shape
    original_h, original_w = original_img_shape
    scale_w = original_w / resized_w
    scale_h = original_h / resized_h
    
    lines = []
    for coco_cls_id in predictions:
        yolo_cls_id = coco_id_to_yolo_id_map.get(coco_cls_id)
        if yolo_cls_id is None: continue
        for bbox in predictions[coco_cls_id]:
            score = bbox[4]
            x1_resized, y1_resized, x2_resized, y2_resized = bbox[:4]
            x1_orig, y1_orig = x1_resized * scale_w, y1_resized * scale_h
            x2_orig, y2_orig = x2_resized * scale_w, y2_resized * scale_h
            x1, x2 = np.clip([x1_orig, x2_orig], 0, original_w - 1)
            y1, y2 = np.clip([y1_orig, y2_orig], 0, original_h - 1)
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 0 or box_h <= 0: continue
            center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
            center_x_norm, w_norm = center_x / original_w, box_w / original_w
            center_y_norm, h_norm = center_y / original_h, box_h / original_h
            lines.append(
                f"{yolo_cls_id} {center_x_norm:.6f} {center_y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}\n")
    
    content = ''.join(lines)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(content)
    return content

# --- 主评估与推理函数 ---
def parallel_inference_and_evaluate(opt, modelPath):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # --- 1. 数据集与模型设置 ---
    print("==> Initializing dataset...")
    dataset = CustomDataset(opt, 'test')
    coco_id_to_yolo_id = dataset.cat_ids
    class_names_list = dataset.class_name[1:]
    print(f"==> Dataset initialized. Found {len(dataset.images)} samples in annotation file.")

    # --- 2. 并行推理逻辑 ---
    FPS_PER_GPU_PROCESS = 13.0
    model_count = math.ceil(opt.fps / FPS_PER_GPU_PROCESS)
    print(f"\nModel path: {modelPath}\nModel: {opt.model_name}\nTarget FPS: {opt.fps}")

    models = load_parallel_models(modelPath, model_count, opt, dataset.num_classes)

    model_file_name = os.path.splitext(os.path.basename(modelPath))[0]
    pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}_parallel_{model_count}', 'yolo_predictions_raw')
    os.makedirs(pred_root_dir, exist_ok=True)
    print(f"Raw predictions will be saved to: {pred_root_dir}")

    # --- 数据预处理 ---
    print("📦 Pre-processing all data...")
    all_data = []
    bar = Bar('Preprocessing', max=len(dataset.images))
    for img_id in dataset.images:
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        full_path = os.path.join(opt.data_dir, file_name)
        
        img = cv2.imread(full_path)
        if img is None:
            # print(f"  - Warning: Cannot read image: {full_path}, skipping.") # 减少不必要的打印
            bar.next()
            continue

        resized_h, resized_w = dataset.default_resolution[1], dataset.default_resolution[0]
        original_h, original_w = img.shape[:2]

        if original_h != resized_h or original_w != resized_w:
            img = cv2.resize(img, (resized_w, resized_h))

        mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

        inp = (img.astype(np.float32) / 255.)
        inp = (inp - mean) / std
        inp = inp.transpose(2, 0, 1)

        seq_inp = np.stack([inp] * opt.seqLen, axis=1)

        meta = {
            'c': np.array([resized_w / 2., resized_h / 2.], dtype=np.float32),
            's': max(resized_h, resized_w) * 1.0,
            'out_height': resized_h // opt.down_ratio,
            'out_width': resized_w // opt.down_ratio,
            'original_height': original_h,
            'original_width': original_w
        }
        all_data.append({'seq_inp': seq_inp, 'meta': meta, 'frame_path': full_path})
        bar.next()
    bar.finish()
    
    total_samples = len(all_data)
    if total_samples == 0:
        print("\nError: No valid image data to process.")
        return {}, 0, 0
    print(f"Total samples to process: {total_samples}")

    # --- 数据分片 ---
    data_splits = np.array_split(all_data, model_count)

    dataset_info = {
        'num_classes': dataset.num_classes,
        'coco_id_to_yolo_id': coco_id_to_yolo_id,
        'default_resolution': dataset.default_resolution,
    }

    # --- 并行处理 ---
    print(f"\n🚀 Starting parallel inference with {model_count} threads...")
    start_time = time.time()
    
    predictions_buffer = defaultdict(list)
    lock = threading.Lock()
    
    tasks = []
    for model_id, (model, data_split) in enumerate(zip(models, data_splits)):
        if len(data_split) > 0:
             tasks.append((model_id, model, data_split.tolist(), opt, dataset_info, predictions_buffer, lock))

    with ThreadPoolExecutor(max_workers=model_count) as executor:
        futures = [executor.submit(inference_worker, task) for task in tasks]
        
        bar = Bar(f'🚀 Parallel Processing ({model_count} threads)', max=total_samples)
        completed_samples = 0
        for future in futures:
            processed_count = future.result()
            completed_samples += processed_count
            bar.goto(completed_samples)
        bar.finish()
        
    end_time = time.time()
    total_inference_time = end_time - start_time
    
    # --- 批量写入文件 ---
    print("\n💾 Writing predictions to files...")
    for video_name, frame_data in predictions_buffer.items():
        save_video_dir = os.path.join(pred_root_dir, video_name)
        os.makedirs(save_video_dir, exist_ok=True)
        for frame_name, content in frame_data:
            save_path = os.path.join(save_video_dir, frame_name + '.txt')
            with open(save_path, 'w') as f:
                f.write(content)

    print(f"\n✅ Total Inference Time: {total_inference_time:.2f} seconds")
    print(f"Average FPS: {total_samples / total_inference_time:.2f}")

    # --- 3. 多置信度评估 ---
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4]
    results_summary = {}

    print("\n📊 Starting multi-confidence evaluation...")
    for i, conf in enumerate(confidence_thresholds):
        print(f"🎯 Evaluating confidence threshold {i + 1}/{len(confidence_thresholds)}: {conf:.2f}")
        filtered_pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}_parallel_{model_count}', f'filtered_preds_conf_{conf:.2f}')
        os.makedirs(filtered_pred_root_dir, exist_ok=True)

        for video_name in os.listdir(pred_root_dir):
            src_video_dir = os.path.join(pred_root_dir, video_name)
            if not os.path.isdir(src_video_dir): continue
            dst_video_dir = os.path.join(filtered_pred_root_dir, video_name)
            os.makedirs(dst_video_dir, exist_ok=True)
            for fname in os.listdir(src_video_dir):
                src_file = os.path.join(src_video_dir, fname)
                if not os.path.isfile(src_file): continue
                with open(src_file, 'r') as f_in, open(os.path.join(dst_video_dir, fname), 'w') as f_out:
                    for line in f_in:
                        if float(line.strip().split()[-1]) >= conf:
                            f_out.write(line)
    
        eval_config = {
            'gt_root': os.path.join(opt.data_dir, 'labels'),
            'pred_root': filtered_pred_root_dir,
            'iou_threshold': opt.iou_thresh,
            'class_names': class_names_list,
        }
        evaluator = DetectionEvaluator(eval_config)
        evaluator.evaluate_all()
        overall_metrics = evaluator.calculate_overall_metrics()

        if 'overall' in overall_metrics:
            metrics = overall_metrics['overall']
            results_summary[conf] = {
                'recall': metrics['recall'], 'precision': metrics['precision'], 'f1': metrics.get('f1', 0.0),
                'false_alarm_rate': metrics['false_alarm_rate'],
                'spatiotemporal_stability': metrics['spatiotemporal_stability'],
                'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn']
            }
        else:
            results_summary[conf] = {'recall': 0, 'precision': 0, 'f1': 0, 'false_alarm_rate': 1.0,
                                     'spatiotemporal_stability': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

    print("✅ Evaluation complete!")
    return results_summary, total_inference_time, model_count

# --- 绘图与结果保存函数 ---
def save_and_plot_results(results, model_name, model_path, model_count):
    model_file_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join('./confidence_analysis_results', f'{model_name}_{model_file_name}_parallel_{model_count}')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'confidence_analysis.json')
    with open(json_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"📁 Full results saved to: {json_path}")
    thresholds = sorted(results.keys())
    if len(thresholds) < 1: return
    recalls = [results[t]['recall'] for t in thresholds]
    fars = [results[t]['false_alarm_rate'] for t in thresholds]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Recall', color=color)
    ax1.plot(thresholds, recalls, marker='o', color=color, label='Recall')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim([0, max(1.0, max(recalls) * 1.1 if recalls else 1.0)])
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('False Alarm Rate', color=color)
    ax2.plot(thresholds, fars, marker='s', linestyle='--', color=color, label='False Alarm Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, max(1.0, max(fars) * 1.1 if fars else 1.0)])
    fig.suptitle(f'Recall & FAR vs. Confidence (Parallel {model_count} threads)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plot_path = os.path.join(output_dir, 'Recall_FAR_vs_Confidence.png')
    plt.savefig(plot_path)
    print(f"📈 Performance curves saved to: {plot_path}")

# --- 主执行块 ---
if __name__ == '__main__':
    opts_parser = opts()
    opt = opts_parser.parse()
    opt = opts_parser.init(opt)

    if opt.load_model == '':
        # 请确保这里的路径是正确的
        modelPath = './checkpoint/DSFNet.pth'
    else:
        modelPath = opt.load_model

    results_summary, total_time, model_count = parallel_inference_and_evaluate(opt, modelPath)

    if results_summary:
        print("\n" + "=" * 95)
        print(f"📊 Confidence Threshold Performance Summary (Parallel {model_count} threads)")
        print("=" * 95)
        print(
            f"{'Conf.':<6} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'Stability':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 95)
        for conf, metrics in sorted(results_summary.items()):
            print(f"{conf:<6.1f} {metrics['recall']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['false_alarm_rate']:<10.4f} {metrics['spatiotemporal_stability']:<12.4f} "
                  f"{metrics['tp']:<8} {metrics['fp']:<8} {metrics['fn']:<8}")
        print("=" * 95)
        print(f"⏱️ Total Inference Time: {total_time:.2f} seconds")
        print(f"🎯 Models Used: {model_count} (based on FPS={opt.fps})")
        print("-" * 95)
        save_and_plot_results(results_summary, opt.model_name, modelPath, model_count)

    print("\n✅ Multi-confidence evaluation finished!")