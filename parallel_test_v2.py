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
import multiprocessing
import math
import time
from functools import partial
from torch.utils.data import Dataset, DataLoader

matplotlib.use('Agg')

from lib.utils.opts import opts
from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from evaluator import DetectionEvaluator
import cv2
from lib.utils.image import get_affine_transform

# --- 推理专用数据集 ---
class InferenceDataset(Dataset):
    def __init__(self, frame_paths, opt, dataset_info):
        self.frame_paths = frame_paths
        self.opt = opt
        self.resized_h, self.resized_w = dataset_info['default_resolution'][1], dataset_info['default_resolution'][0]
        self.mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, index):
        frame_path = self.frame_paths[index]
        img = cv2.imread(frame_path)
        if img is None:
            print(f"  - 警告: 无法读取图像: {frame_path}，将返回空张量。")
            return torch.zeros(3, self.opt.seqLen, self.resized_h, self.resized_w), {}, frame_path
        
        original_h, original_w = img.shape[:2]
        
        if original_h != self.resized_h or original_w != self.resized_w:
            img = cv2.resize(img, (self.resized_w, self.resized_h))

        inp = (img.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        seq_inp = np.stack([inp] * self.opt.seqLen, axis=1)

        meta = {
            'c': np.array([self.resized_w / 2., self.resized_h / 2.], dtype=np.float32),
            's': max(self.resized_h, self.resized_w) * 1.0,
            'out_height': self.resized_h // self.opt.down_ratio,
            'out_width': self.resized_w // self.opt.down_ratio,
            'original_height': original_h,
            'original_width': original_w
        }
        return torch.from_numpy(seq_inp), meta, frame_path

# --- 并行推理工作函数 ---
def inference_worker(args):
    """
    在指定GPU上对一部分帧进行推理的工作进程 (实现手动批处理)。
    """
    gpu_id, frame_paths, opt, model_path, dataset_info = args
    
    pid = os.getpid()
    print(f"[Worker PID: {pid}]  Assigned to GPU: {gpu_id}, processing {len(frame_paths)} frames.")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda')

    model = get_det_net({'hm': dataset_info['num_classes'], 'wh': 2, 'reg': 2}, opt.model_name)
    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()

    resized_h, resized_w = dataset_info['default_resolution'][1], dataset_info['default_resolution'][0]
    mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

    # ==================== [核心修改: 手动实现批处理] ====================
    
    # 1. 首先，预处理所有分配给这个worker的帧
    preprocessed_data = []
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is None:
            print(f"  - 警告: 无法读取图像: {frame_path}，跳过。")
            continue
        
        original_h, original_w = img.shape[:2]

        if original_h != resized_h or original_w != resized_w:
            img = cv2.resize(img, (resized_w, resized_h))

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
        preprocessed_data.append({'seq_inp': seq_inp, 'meta': meta, 'frame_path': frame_path})

    # 2. 将预处理好的数据分批 (batching)
    batch_size = opt.test_batch_size
    num_batches = math.ceil(len(preprocessed_data) / batch_size)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(preprocessed_data))
        current_batch = preprocessed_data[start_idx:end_idx]
        
        if not current_batch:
            continue

        # 将批次数据堆叠成一个大的张量
        seq_inp_batch = np.array([item['seq_inp'] for item in current_batch])
        image_batch = torch.from_numpy(seq_inp_batch).to(device) # (B, C, T, H, W)
        
        # 模型推理
        dets_raw_batch = process(model, image_batch, opt)
        
        # 3. 逐个处理批次中的结果
        for j, item in enumerate(current_batch):
            dets_raw = dets_raw_batch[j:j+1]
            meta = item['meta']
            frame_path = item['frame_path']
            
            dets = post_process(dets_raw, meta, dataset_info['num_classes'])
            
            path_parts = frame_path.replace('\\', '/').split('/')
            video_name = path_parts[-2]
            frame_name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]

            save_video_dir = os.path.join(dataset_info['pred_root_dir'], video_name)
            os.makedirs(save_video_dir, exist_ok=True)
            save_path = os.path.join(save_video_dir, frame_name_no_ext + '.txt')

            save_predictions_as_yolo(
                predictions=dets,
                resized_shape=(resized_h, resized_w),
                original_img_shape=(meta['original_height'], meta['original_width']),
                save_path=save_path,
                coco_id_to_yolo_id_map=dataset_info['coco_id_to_yolo_id']
            )
            
    # ========================================================================
    return True

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
    
    with open(save_path, 'w') as f:
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
                f.write(
                    f"{yolo_cls_id} {center_x_norm:.6f} {center_y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}\n")

# --- 主评估与推理函数 ---
def parallel_inference_and_evaluate(opt, modelPath):
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # --- 1. 数据集与模型设置 (主进程) ---
    print("==> Initializing dataset...")
    dataset = CustomDataset(opt, 'test')
    coco_id_to_yolo_id = dataset.cat_ids
    class_names_list = dataset.class_name[1:]
    print(f"==> Dataset initialized. Found {len(dataset.images)} samples in annotation file.")

    model_file_name = os.path.splitext(os.path.basename(modelPath))[0]
    pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}', 'yolo_predictions_raw')
    os.makedirs(pred_root_dir, exist_ok=True)
    print(f"Raw predictions will be saved to: {pred_root_dir}")

    # --- 2. 并行推理逻辑 ---
    all_frame_paths = []
    worker_tasks = []

    print("正在从 annotation JSON 文件收集待处理的帧列表...")
    for img_id in dataset.images:
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']
        full_path = os.path.join(opt.data_dir, file_name)
        all_frame_paths.append(full_path)
    
    if not all_frame_paths:
        print("\n错误: 从 annotation JSON 文件中未能加载任何图像路径。")
        return {}, 0
    
    # ==================== [核心修复: 修正进程数计算和GPU分配逻辑] ====================
    FPS_PER_GPU_PROCESS = 13.0
    fps = opt.fps
    
    # 1. 直接计算理想的进程数，不再受限于可用GPU数量
    num_processes_needed = math.ceil(fps / FPS_PER_GPU_PROCESS)
    
    num_available_gpus = len(opt.gpus)
    print(f"总共找到 {len(all_frame_paths)} 帧图像。")
    print(f"根据帧率 {fps} FPS，理想进程数为: {num_processes_needed}")
    print(f"可用 GPU 数量为: {num_available_gpus}。将在这些 GPU 上循环启动进程。")
    
    frame_chunks = np.array_split(all_frame_paths, num_processes_needed)

    dataset_info = {
        'num_classes': dataset.num_classes,
        'coco_id_to_yolo_id': coco_id_to_yolo_id,
        'default_resolution': dataset.default_resolution,
        'pred_root_dir': pred_root_dir,
    }

    # 2. 创建任务，并使用模运算循环分配GPU
    for i in range(num_processes_needed):
        # 使用 i % num_available_gpus 来循环选择GPU
        gpu_id = opt.gpus[i % num_available_gpus]
        chunk = frame_chunks[i].tolist()
        if not chunk: continue
        worker_tasks.append((gpu_id, chunk, opt, modelPath, dataset_info))
    # ==============================================================================

    if not worker_tasks:
        print("\n错误: 未能创建任何有效的推理任务。")
        return {}, 0

    start_time = time.time()
    # 进程池的大小等于计算出的理想进程数
    with multiprocessing.Pool(processes=num_processes_needed) as pool:
        num_tasks = len(worker_tasks)
        bar = Bar('🚀 Inference Phase', max=num_tasks)
        for _ in pool.imap_unordered(inference_worker, worker_tasks):
            bar.next()
        bar.finish()

    end_time = time.time()
    total_inference_time = end_time - start_time
    print(f"\n✅ Total Inference Time: {total_inference_time:.2f} seconds")

    # --- 3. 多置信度评估 ---
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4]
    results_summary = {}

    print("\n📊 Starting multi-confidence evaluation...")
    for i, conf in enumerate(confidence_thresholds):
        print(f"🎯 Evaluating confidence threshold {i + 1}/{len(confidence_thresholds)}: {conf:.2f}")
        filtered_pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}', f'filtered_preds_conf_{conf:.2f}')
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
    return results_summary, total_inference_time

# --- 绘图与结果保存函数 ---
def save_and_plot_results(results, model_name, model_path):
    model_file_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join('./confidence_analysis_results', f'{model_name}_{model_file_name}')
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
    fig.suptitle('Recall and False Alarm Rate vs. Confidence Threshold', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plot_path = os.path.join(output_dir, 'Recall_FAR_vs_Confidence.png')
    plt.savefig(plot_path)
    print(f"📈 Performance curves saved to: {plot_path}")

# --- 主执行块 ---
if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn', force=True)
    opts_parser = opts()
    opt = opts_parser.parse()
    opt = opts_parser.init(opt)

    if opt.load_model == '':
        modelPath = './checkpoint/DSFNet.pth'
    else:
        modelPath = opt.load_model

    results_summary, total_time = parallel_inference_and_evaluate(opt, modelPath)

    if results_summary:
        print("\n" + "=" * 95)
        print("📊 Confidence Threshold Performance Summary")
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
        print("-" * 95)
        save_and_plot_results(results_summary, opt.model_name, modelPath)

    print("\n✅ Multi-confidence evaluation finished!")