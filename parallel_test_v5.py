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

# --- 所有辅助函数 (inference_thread, gpu_worker_process, etc.) ---
# --- 与之前的版本完全相同，无需改动。                               ---
def inference_thread(args):
    thread_id, model, device, frame_path, opt, dataset_info, predictions_buffer, lock = args
    img = cv2.imread(frame_path)
    if img is None: return 0
    resized_h, resized_w = dataset_info['default_resolution'][1], dataset_info['default_resolution'][0]
    original_h, original_w = img.shape[:2]
    img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR) if (original_h, original_w) != (resized_h, resized_w) else img
    mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)
    inp = (((img_resized.astype(np.float32) / 255.) - mean) / std).transpose(2, 0, 1)
    seq_inp = np.stack([inp] * opt.seqLen, axis=1)
    meta = {'c': np.array([resized_w/2., resized_h/2.]), 's': max(resized_h, resized_w)*1.0, 'out_height': resized_h//opt.down_ratio, 'out_width': resized_w//opt.down_ratio, 'original_height': original_h, 'original_width': original_w}
    image_tensor = torch.from_numpy(np.array([seq_inp])).to(device)
    with torch.no_grad():
        output = model(image_tensor)[-1]
        hm, wh, reg = output['hm'].sigmoid_(), output['wh'], output.get('reg', None)
        dets_raw = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    dets = post_process(dets_raw, meta, dataset_info['num_classes'])
    pred_content = get_yolo_format_string(dets, (resized_h, resized_w), (original_h, original_w), dataset_info['coco_id_to_yolo_id'])
    video_name = frame_path.replace('\\', '/').split('/')[-2]
    frame_name_no_ext = os.path.splitext(os.path.basename(frame_path))[0]
    with lock:
        predictions_buffer[video_name].append((frame_name_no_ext, pred_content))
    return 1

def gpu_worker_process(args):
    process_index, gpu_id, frame_path_chunk, opt, model_path, dataset_info, num_threads_in_this_process = args
    pid = os.getpid()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda')
    print(f"[Worker PID:{pid}] Attached to GPU:{gpu_id}. Manages {num_threads_in_this_process} threads for {len(frame_path_chunk)} images.")
    model = load_model_for_worker(model_path, opt, device, dataset_info['num_classes'], process_index)
    buffer = defaultdict(list)
    lock = threading.Lock()
    tasks = [(i, model, device, path, opt, dataset_info, buffer, lock) for i, path in enumerate(frame_path_chunk)]
    with ThreadPoolExecutor(max_workers=num_threads_in_this_process) as executor:
        list(executor.map(inference_thread, tasks))
    pred_root_dir = dataset_info['pred_root_dir']
    for video, data in buffer.items():
        save_dir = os.path.join(pred_root_dir, video)
        os.makedirs(save_dir, exist_ok=True)
        for frame, content in data:
            with open(os.path.join(save_dir, f'{frame}.txt'), 'w') as f: f.write(content)
    return True

def load_model_for_worker(model_path, opt, device, dataset_num_classes, process_index):
    model = get_det_net({'hm': dataset_num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
    checkpoint = torch.load(model_path, map_location=device)
    is_multi_model = isinstance(checkpoint, dict) and any(k.startswith('model0.') for k in checkpoint.keys())
    if is_multi_model:
        keys = [k for k in checkpoint.keys() if k.startswith('model')]
        num_models = len(set([k.split('.')[0] for k in keys]))
        idx = process_index % num_models
        prefix = f'model{idx}.'
        state_dict = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
        if state_dict: model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    return model.to(device).eval()

def post_process(dets, meta, num_classes):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5) if len(dets[0][j]) > 0 else np.empty([0, 5], dtype=np.float32)
    return dets[0]

def get_yolo_format_string(predictions, resized_shape, original_img_shape, coco_id_to_yolo_id_map):
    h, w = resized_shape
    oh, ow = original_img_shape
    sw, sh = ow / w, oh / h
    lines = []
    for cid, bboxes in predictions.items():
        yid = coco_id_to_yolo_id_map.get(cid)
        if yid is None: continue
        for bbox in bboxes:
            s, (x1, y1, x2, y2) = bbox[4], bbox[:4]
            x1o, y1o, x2o, y2o = x1 * sw, y1 * sh, x2 * sw, y2 * sh
            x1c, x2c = np.clip([x1o, x2o], 0, ow - 1)
            y1c, y2c = np.clip([y1o, y2o], 0, oh - 1)
            bw, bh = x2c - x1c, y2c - y1c
            if bw <= 0 or bh <= 0: continue
            cx, cy = x1c + bw / 2, y1c + bh / 2
            lines.append(f"{yid} {cx/ow:.6f} {cy/oh:.6f} {bw/ow:.6f} {bh/oh:.6f} {s:.6f}\n")
    return ''.join(lines)


# --- 主评估与推理函数 ---
def parallel_inference_and_evaluate(opt, modelPath):
    # --- 1. 获取图像路径列表 ---
    print("==> Initializing dataset and gathering image paths...")
    dataset = CustomDataset(opt, 'test')
    all_frame_paths = [os.path.join(opt.data_dir, dataset.coco.loadImgs(ids=[img_id])[0]['file_name']) for img_id in dataset.images]
    total_samples = len(all_frame_paths)
    print(f"==> Found {total_samples} images to process.")
    if total_samples == 0: return {}, 0, {}

    # --- 2. 实现您的三层决策模型 ---
    FPS_PER_EXPERT = 13.0
    num_gpus = len(opt.gpus)
    
    total_experts_needed = math.ceil(opt.fps / FPS_PER_EXPERT)
    experts_per_gpu = math.ceil(total_experts_needed / num_gpus)
    processes_per_gpu = math.ceil(experts_per_gpu / opt.max_threads_per_process)
    
    print("\n🚀 Fully Customized Parallelism Configuration:")
    # ... (打印配置信息的代码不变)
    print(f"   - Target FPS: {opt.fps}")
    print(f"   - GPUs to use: {opt.gpus} ({num_gpus} cards)")
    print(f"   - Max Threads per Process (user set): {opt.max_threads_per_process}")
    print(f"   - --------------------------------------------------")
    print(f"   - Layer 1 -> Total Experts/Threads needed: {total_experts_needed}")
    print(f"   - Layer 2 -> Experts/Threads per GPU: {experts_per_gpu}")
    print(f"   - Layer 3 -> Processes per GPU: {processes_per_gpu}")
    if processes_per_gpu > 1:
        print("\n   🔥🔥🔥 WARNING: `processes_per_gpu` > 1 🔥🔥🔥")
        print(f"   This will launch {processes_per_gpu} separate processes on EACH GPU.")
        print(f"   This will multiply VRAM usage and may cause CUDA Out of Memory errors.")
        print(f"   Proceed with caution!")


    # --- 3. 创建最终任务列表 ---
    total_processes = processes_per_gpu * num_gpus
    all_path_chunks = np.array_split(all_frame_paths, total_experts_needed)
    
    model_file_name = os.path.splitext(os.path.basename(modelPath))[0]
    pred_root_dir = os.path.join('./results', f'{model_file_name}_custom_hybrid', 'yolo_predictions_raw')
    os.makedirs(pred_root_dir, exist_ok=True)
    
    dataset_info = {
        'num_classes': dataset.num_classes, 
        'coco_id_to_yolo_id': dataset.cat_ids,
        'default_resolution': dataset.default_resolution, 
        'pred_root_dir': pred_root_dir
    }

    tasks = []
    expert_chunk_counter = 0
    for i in range(total_processes):
        gpu_id = opt.gpus[i % num_gpus]
        
        experts_left_on_this_gpu = experts_per_gpu
        # This logic is a bit tricky, let's simplify for clarity and correctness
        # Correctly calculate threads for this specific process
        experts_assigned_to_this_gpu = np.array_split(range(experts_per_gpu), processes_per_gpu)[i // num_gpus]
        num_threads_in_this_process = len(experts_assigned_to_this_gpu)

        process_task_chunks = all_path_chunks[expert_chunk_counter : expert_chunk_counter + num_threads_in_this_process]
        process_frame_paths = [path for chunk in process_task_chunks for path in chunk]
        expert_chunk_counter += num_threads_in_this_process
        
        if not process_frame_paths: continue
        
        tasks.append((i, gpu_id, process_frame_paths, opt, modelPath, dataset_info, num_threads_in_this_process))

    # --- 4. 启动进程池 ---
    start_time = time.time()
    with multiprocessing.Pool(processes=total_processes) as pool:
        bar = Bar('Custom Hybrid Inference', max=len(tasks))
        for _ in pool.imap_unordered(gpu_worker_process, tasks):
            bar.next()
        bar.finish()
    total_inference_time = time.time() - start_time
    print(f"\n✅ Inference Complete! Time: {total_inference_time:.2f}s, Avg FPS: {total_samples / total_inference_time:.2f}")

    # --- 5. 评估阶段 (已恢复) ---
    print("\n📊 Starting multi-confidence evaluation...")
    results_summary = {}
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4]
    
    for conf in confidence_thresholds:
        print(f"🎯 Evaluating confidence threshold: {conf:.2f}")
        # 创建用于存放筛选后结果的目录
        filtered_pred_root_dir = os.path.join(os.path.dirname(pred_root_dir), f'filtered_preds_conf_{conf:.2f}')
        os.makedirs(filtered_pred_root_dir, exist_ok=True)

        # 遍历原始预测结果并根据置信度筛选
        for video_name in os.listdir(pred_root_dir):
            src_video_dir = os.path.join(pred_root_dir, video_name)
            if not os.path.isdir(src_video_dir): continue
            
            dst_video_dir = os.path.join(filtered_pred_root_dir, video_name)
            os.makedirs(dst_video_dir, exist_ok=True)
            
            for fname in os.listdir(src_video_dir):
                src_file = os.path.join(src_video_dir, fname)
                dst_file = os.path.join(dst_video_dir, fname)
                with open(src_file, 'r') as f_in, open(dst_file, 'w') as f_out:
                    for line in f_in:
                        # 假设置信度是每行的最后一项
                        if float(line.strip().split()[-1]) >= conf:
                            f_out.write(line)

        # 调用评估器进行评估
        eval_config = {
            'gt_root': os.path.join(opt.data_dir, 'labels'),
            'pred_root': filtered_pred_root_dir,
            'iou_threshold': opt.iou_thresh,
            'class_names': dataset.class_name[1:]
        }
        evaluator = DetectionEvaluator(eval_config)
        evaluator.evaluate_all()
        overall_metrics = evaluator.calculate_overall_metrics()

        # 保存评估结果
        if 'overall' in overall_metrics:
            results_summary[conf] = overall_metrics['overall']
        else:
            # 提供一个默认的空结果，以防评估失败
            results_summary[conf] = {'recall': 0, 'precision': 0, 'f1': 0, 'false_alarm_rate': 1.0, 
                                     'spatiotemporal_stability': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

    print("✅ Evaluation complete!")
    return results_summary, total_inference_time, {"processes": total_processes, "threads_per_process_max": opt.max_threads_per_process}

# --- 主执行块 ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    opts_parser = opts()
    opts_parser.parser.add_argument('--max_threads_per_process', type=int, default=4, 
                                    help='Max number of threads a single process can manage. Controls process granularity.')
    opt = opts_parser.parse()
    opt = opts_parser.init(opt)
    modelPath = opt.load_model if opt.load_model != '' else './checkpoint/DSFNet.pth'
    
    results_summary, total_time, config = parallel_inference_and_evaluate(opt, modelPath)
    
    if results_summary:
        print("\n" + "=" * 95)
        print(f"📊 Final Performance Summary")
        print(f"   (Total Processes: {config['processes']}, Max Threads/Process: {config['threads_per_process_max']})")
        print("=" * 95)
        print(f"{'Conf.':<6} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'Stability':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 95)
        for conf, metrics in sorted(results_summary.items()):
            recall = metrics.get('recall', 0)
            precision = metrics.get('precision', 0)
            far = metrics.get('false_alarm_rate', 1.0)
            stability = metrics.get('spatiotemporal_stability', 0)
            tp, fp, fn = metrics.get('tp', 0), metrics.get('fp', 0), metrics.get('fn', 0)
            print(f"{conf:<6.1f} {recall:<10.4f} {precision:<10.4f} {far:<10.4f} {stability:<12.4f} {tp:<8} {fp:<8} {fn:<8}")
        print("=" * 95)

    print(f"\n⏱️ Total Inference Time: {total_time:.2f} seconds")
    print("\n✅ All finished!")