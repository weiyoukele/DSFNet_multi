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
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

matplotlib.use('Agg')

from lib.utils.opts import opts
from lib.models.stNet import get_det_net
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from evaluator import DetectionEvaluator
import cv2

# --- 线程级推理函数 (现在包含预处理) ---
def inference_thread(args):
    """
    在单个线程中，为一个图像路径执行完整的“预处理 -> 推理 -> 后处理”流程。
    """
    thread_id, model, device, frame_path, opt, dataset_info, predictions_buffer, lock = args
    
    # 1. 线程内进行图像读取和预处理
    img = cv2.imread(frame_path)
    if img is None:
        # print(f"Warning: Failed to read image {frame_path}")
        return 0 # 返回处理了0个项目

    resized_h, resized_w = dataset_info['default_resolution'][1], dataset_info['default_resolution'][0]
    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR) if (original_h, original_w) != (resized_h, resized_w) else img
    
    mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)
    inp = (((img_resized.astype(np.float32) / 255.) - mean) / std).transpose(2, 0, 1)
    seq_inp = np.stack([inp] * opt.seqLen, axis=1)

    meta = {'c': np.array([resized_w/2., resized_h/2.]), 's': max(resized_h, resized_w)*1.0,
            'out_height': resized_h//opt.down_ratio, 'out_width': resized_w//opt.down_ratio,
            'original_height': original_h, 'original_width': original_w}
    
    image_tensor = torch.from_numpy(np.array([seq_inp])).to(device)

    # 2. 使用共享的模型进行推理
    with torch.no_grad():
        output = model(image_tensor)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output.get('reg', None)
        dets_raw = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    
    dets = post_process(dets_raw, meta, dataset_info['num_classes'])
    
    # 3. 生成YOLO格式的字符串并暂存
    pred_content = get_yolo_format_string(
        predictions=dets,
        resized_shape=(resized_h, resized_w),
        original_img_shape=(original_h, original_w),
        coco_id_to_yolo_id_map=dataset_info['coco_id_to_yolo_id']
    )
    
    path_parts = frame_path.replace('\\', '/').split('/')
    video_name = path_parts[-2]
    frame_name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]
    
    # 使用锁安全地写入共享缓冲区
    with lock:
        predictions_buffer[video_name].append((frame_name_no_ext, pred_content))
    
    return 1 # 返回处理了1个项目

# --- GPU级工作进程函数 ---
def gpu_worker_process(args):
    """
    绑定到一张GPU，加载一个模型，并管理一个线程池在该GPU上并行处理图像路径。
    """
    process_index, gpu_id, frame_path_chunk, opt, model_path, dataset_info, threads_per_gpu = args
    pid = os.getpid()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda')
    
    print(f"[GPU Worker PID: {pid}]  Attached to GPU: {gpu_id}. Manages {threads_per_gpu} threads for {len(frame_path_chunk)} images.")

    model = load_model_for_worker(model_path, opt, device, dataset_info['num_classes'], process_index)

    process_predictions_buffer = defaultdict(list)
    lock = threading.Lock()
    
    # 创建任务列表，每个任务是一个图像路径
    thread_tasks = [(i, model, device, frame_path, opt, dataset_info, process_predictions_buffer, lock) 
                    for i, frame_path in enumerate(frame_path_chunk)]

    with ThreadPoolExecutor(max_workers=threads_per_gpu) as executor:
        # 使用imap来获取进度，但我们只关心完成，所以用list消耗掉它
        list(executor.map(inference_thread, thread_tasks))

    # 所有线程完成后，由主进程将结果批量写入文件
    pred_root_dir = dataset_info['pred_root_dir']
    for video_name, frame_data in process_predictions_buffer.items():
        save_video_dir = os.path.join(pred_root_dir, video_name)
        os.makedirs(save_video_dir, exist_ok=True)
        for frame_name, content in frame_data:
            save_path = os.path.join(save_video_dir, frame_name + '.txt')
            with open(save_path, 'w') as f:
                f.write(content)
    
    return True

# --- 辅助函数 (基本不变) ---
def load_model_for_worker(model_path, opt, device, dataset_num_classes, process_index):
    # ... (此函数与上一版本完全相同)
    model = get_det_net({'hm': dataset_num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
    checkpoint = torch.load(model_path, map_location=device)
    is_multi_model_checkpoint = isinstance(checkpoint, dict) and any(key.startswith('model0.') for key in checkpoint.keys())
    if is_multi_model_checkpoint:
        model_keys = [key for key in checkpoint.keys() if key.startswith('model')]
        num_models_in_file = len(set([key.split('.')[0] for key in model_keys]))
        model_to_load_idx = process_index % num_models_in_file
        model_prefix = f'model{model_to_load_idx}.'
        model_state_dict = {key[len(model_prefix):]: value for key, value in checkpoint.items() if key.startswith(model_prefix)}
        if model_state_dict: model.load_state_dict(model_state_dict)
    else:
        base_model_state = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(base_model_state)
    model = model.to(device)
    model.eval()
    return model

def post_process(dets, meta, num_classes):
    # ... (此函数与上一版本完全相同)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5) if len(dets[0][j]) > 0 else np.empty([0, 5], dtype=np.float32)
    return dets[0]

def get_yolo_format_string(predictions, resized_shape, original_img_shape, coco_id_to_yolo_id_map):
    # ... (此函数与上一版本完全相同)
    resized_h, resized_w = resized_shape
    original_h, original_w = original_img_shape
    scale_w, scale_h = original_w / resized_w, original_h / resized_h
    lines = []
    for coco_cls_id, bboxes in predictions.items():
        yolo_cls_id = coco_id_to_yolo_id_map.get(coco_cls_id)
        if yolo_cls_id is None: continue
        for bbox in bboxes:
            score = bbox[4]
            x1, y1, x2, y2 = bbox[:4]
            x1_orig, y1_orig, x2_orig, y2_orig = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
            x1c, x2c = np.clip([x1_orig, x2_orig], 0, original_w - 1)
            y1c, y2c = np.clip([y1_orig, y2_orig], 0, original_h - 1)
            box_w, box_h = x2c - x1c, y2c - y1c
            if box_w <= 0 or box_h <= 0: continue
            center_x, center_y = x1c + box_w / 2, y1c + box_h / 2
            lines.append(f"{yolo_cls_id} {center_x / original_w:.6f} {center_y / original_h:.6f} {box_w / original_w:.6f} {box_h / original_h:.6f} {score:.6f}\n")
    return ''.join(lines)


# --- 主评估与推理函数 (重大修改) ---
def parallel_inference_and_evaluate(opt, modelPath):
    # --- 1. 获取图像路径列表 (不再预处理!) ---
    print("==> Initializing dataset and gathering image paths...")
    dataset = CustomDataset(opt, 'test')
    
    all_frame_paths = []
    for img_id in dataset.images:
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        full_path = os.path.join(opt.data_dir, img_info['file_name'])
        all_frame_paths.append(full_path)
    
    print(f"==> Found {len(all_frame_paths)} images to process.")
    
    total_samples = len(all_frame_paths)
    if total_samples == 0:
        print("\nError: No images found."); return {}, 0, 0

    # --- 2. 设置混合并行任务 ---
    num_gpus = len(opt.gpus)
    threads_per_gpu = getattr(opt, 'threads_per_gpu', 4)
    print(f"\n🚀 Starting Hybrid Parallel Inference:")
    print(f"   - GPUs to use: {opt.gpus} ({num_gpus} cards)")
    print(f"   - Processes: {num_gpus} (1 per GPU)")
    print(f"   - Threads per Process: {threads_per_gpu}")

    model_file_name = os.path.splitext(os.path.basename(modelPath))[0]
    pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}_hybrid_parallel', 'yolo_predictions_raw')
    os.makedirs(pred_root_dir, exist_ok=True)
    print(f"   - Predictions will be saved to: {pred_root_dir}")

    # 将路径列表分成N块，N=GPU数量
    path_chunks_for_processes = np.array_split(all_frame_paths, num_gpus)

    dataset_info = {
        'num_classes': dataset.num_classes,
        'coco_id_to_yolo_id': dataset.cat_ids,
        'default_resolution': dataset.default_resolution,
        'pred_root_dir': pred_root_dir
    }
    
    tasks = [(i, opt.gpus[i], path_chunks_for_processes[i].tolist(), opt, modelPath, dataset_info, threads_per_gpu) 
             for i in range(num_gpus) if len(path_chunks_for_processes[i]) > 0]

    # --- 3. 启动进程池 ---
    start_time = time.time()
    with multiprocessing.Pool(processes=num_gpus) as pool:
        bar = Bar('Hybrid Inference on GPUs', max=len(tasks))
        # imap_unordered会立即返回一个迭代器，当我们消耗它时，进程才真正开始阻塞
        for _ in pool.imap_unordered(gpu_worker_process, tasks):
            bar.next()
        bar.finish()
    
    total_inference_time = time.time() - start_time
    print(f"\n✅ Hybrid Inference Complete! Time: {total_inference_time:.2f}s")
    print(f"   - Average FPS: {total_samples / total_inference_time:.2f}")

    # --- 4. 评估阶段 (不变) ---
    print("\n📊 Starting multi-confidence evaluation...")
    # ... (评估代码与之前版本完全相同) ...
    results_summary = {}
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4]
    for conf in confidence_thresholds:
        # print(f"🎯 Evaluating confidence threshold: {conf:.2f}")
        filtered_pred_root_dir = os.path.join(os.path.dirname(pred_root_dir), f'filtered_preds_conf_{conf:.2f}')
        os.makedirs(filtered_pred_root_dir, exist_ok=True)
        for video_name in os.listdir(pred_root_dir):
            src_video_dir, dst_video_dir = os.path.join(pred_root_dir, video_name), os.path.join(filtered_pred_root_dir, video_name)
            if not os.path.isdir(src_video_dir): continue
            os.makedirs(dst_video_dir, exist_ok=True)
            for fname in os.listdir(src_video_dir):
                with open(os.path.join(src_video_dir, fname), 'r') as f_in, open(os.path.join(dst_video_dir, fname), 'w') as f_out:
                    [f_out.write(line) for line in f_in if float(line.strip().split()[-1]) >= conf]
        eval_config = {'gt_root': os.path.join(opt.data_dir, 'labels'), 'pred_root': filtered_pred_root_dir, 'iou_threshold': opt.iou_thresh, 'class_names': dataset.class_name[1:]}
        evaluator = DetectionEvaluator(eval_config)
        evaluator.evaluate_all()
        overall_metrics = evaluator.calculate_overall_metrics()
        results_summary[conf] = overall_metrics.get('overall', {'recall': 0, 'precision': 0, 'f1': 0, 'false_alarm_rate': 1.0, 'spatiotemporal_stability': 0.0, 'tp': 0, 'fp': 0, 'fn': 0})
    print("✅ Evaluation complete!")
    return results_summary, total_inference_time, (num_gpus, threads_per_gpu)

# --- 主执行块 (与上一版相同) ---
if __name__ == '__main__':
    # ... (代码与上一版本完全相同) ...
    multiprocessing.set_start_method('spawn', force=True)
    opts_parser = opts()
    opts_parser.parser.add_argument('--threads_per_gpu', type=int, default=4, help='Number of worker threads per GPU process.')
    opt = opts_parser.parse()
    opt = opts_parser.init(opt)
    modelPath = opt.load_model if opt.load_model != '' else './checkpoint/DSFNet.pth'
    results_summary, total_time, (num_gpus, threads_per_gpu) = parallel_inference_and_evaluate(opt, modelPath)
    if results_summary:
        print("\n" + "=" * 95)
        print(f"📊 Hybrid Parallel Performance Summary (Processes: {num_gpus}, Threads/Process: {threads_per_gpu} on GPUs: {opt.gpus})")
        print("=" * 95)
        print(f"{'Conf.':<6} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'Stability':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 95)
        for conf, metrics in sorted(results_summary.items()):
            recall, precision, far = metrics.get('recall', 0), metrics.get('precision', 0), metrics.get('false_alarm_rate', 1.0)
            stability, tp, fp, fn = metrics.get('spatiotemporal_stability', 0), metrics.get('tp', 0), metrics.get('fp', 0), metrics.get('fn', 0)
            print(f"{conf:<6.1f} {recall:<10.4f} {precision:<10.4f} {far:<10.4f} {stability:<12.4f} {tp:<8} {fp:<8} {fn:<8}")
        print("=" * 95)
        print(f"⏱️ Total Inference Time: {total_time:.2f} seconds")
        print(f"🎯 Config: {num_gpus} GPU(s) x {threads_per_gpu} Threads/GPU")
        print("-" * 95)
    print("\n✅ Hybrid parallel evaluation finished!")