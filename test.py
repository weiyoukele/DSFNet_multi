from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import json
from progress.bar import Bar
import matplotlib.pyplot as plt
import matplotlib  # 确保在无头服务器上正常工作
import time      # <--- [新增] 导入 time 模块

matplotlib.use('Agg')

from lib.utils.opts import opts
from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from evaluator import DetectionEvaluator


# --- 辅助函数 (保持不变) ---
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
                if box_w <= 0 or box_h <= 0:
                    continue
                center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
                center_x_norm = center_x / original_w
                w_norm = box_w / original_w
                center_y_norm = center_y / original_h
                h_norm = box_h / original_h
                f.write(
                    f"{yolo_cls_id} {center_x_norm:.6f} {center_y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}\n")


# --- 评估主函数 ---
def test_and_evaluate_multi_confidence(opt, modelPath):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print(f"📍 Model path: {modelPath}\nModel: {opt.model_name}")

    dataset = CustomDataset(opt, 'test')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
    print(f"Loading model from: {modelPath}")
    model = load_model(model, modelPath)
    model = model.to(opt.device)
    model.eval()

    coco_id_to_yolo_id = dataset.cat_ids
    class_names_list = dataset.class_name[1:]

    # --- 2. 推理阶段 ---
    model_file_name = os.path.splitext(os.path.basename(modelPath))[0]
    pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}', 'yolo_predictions_raw')
    os.makedirs(pred_root_dir, exist_ok=True)
    print(f"Raw predictions (conf=0.0) will be saved to: {pred_root_dir}")

    bar = Bar(f'🚀 Inference Phase', max=len(data_loader))
    
    # ==================== [核心修改: 添加计时] ====================
    start_time = time.time()
    # ==========================================================

    for ind, (img_id, batch) in enumerate(data_loader):
        image = batch['input'].to(opt.device)
        meta = batch['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}

        original_h, original_w = meta['original_height'], meta['original_width']
        resized_h, resized_w = image.shape[3], image.shape[4]
        file_rel_path = dataset.coco.loadImgs(ids=[img_id.item()])[0]['file_name']

        dets_raw = process(model, image, opt)
        dets = post_process(dets_raw, meta, dataset.num_classes)

        path_parts = file_rel_path.replace('\\', '/').split('/')
        video_name = path_parts[-2] if len(path_parts) > 1 else 'video_root'
        frame_name_no_ext = os.path.splitext(os.path.basename(file_rel_path))[0]
        save_video_dir = os.path.join(pred_root_dir, video_name)
        os.makedirs(save_video_dir, exist_ok=True)
        save_path = os.path.join(save_video_dir, frame_name_no_ext + '.txt')

        save_predictions_as_yolo(
            predictions=dets,
            resized_shape=(resized_h, resized_w),
            original_img_shape=(original_h, original_w),
            save_path=save_path,
            coco_id_to_yolo_id_map=coco_id_to_yolo_id
        )

        bar.next()
    
    # ==================== [核心修改: 结束计时并计算] ====================
    end_time = time.time()
    total_inference_time = end_time - start_time
    # ===============================================================

    bar.finish()
    print(f"✅ Inference complete!")

    # --- 3. 多置信度评估 ---
    confidence_thresholds = [ 0.1, 0.2, 0.3, 0.4 ]
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
                src_file_path = os.path.join(src_video_dir, fname)
                if not os.path.isfile(src_file_path): continue
                with open(src_file_path, 'r') as f_in, open(os.path.join(dst_video_dir, fname), 'w') as f_out:
                    for line in f_in:
                        if float(line.strip().split()[-1]) >= conf: f_out.write(line)

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
    # ==================== [核心修改: 返回时间] ====================
    return results_summary, total_inference_time
    # ==========================================================

# --- [绘图和主执行块] ---
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
    ax1.set_xlabel('Confidence Threshold');
    ax1.set_ylabel('Recall', color=color)
    ax1.plot(thresholds, recalls, marker='o', color=color, label='Recall')
    ax1.tick_params(axis='y', labelcolor=color);
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
    lines, labels = ax1.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plot_path = os.path.join(output_dir, 'Recall_FAR_vs_Confidence.png')
    plt.savefig(plot_path)
    print(f"📈 Performance curves saved to: {plot_path}")


if __name__ == '__main__':
    opts_parser = opts()
    opt = opts_parser.parse()

    opt = opts_parser.init(opt)

    if opt.load_model == '':
        modelPath = './checkpoint/DSFNet.pth'
    else:
        modelPath = opt.load_model

    # ==================== [核心修改: 接收并打印时间] ====================
    results_summary, total_time = test_and_evaluate_multi_confidence(opt, modelPath)

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
        # 在总结表格后打印推理时间
        print(f"⏱️ Total Inference Time: {total_time:.2f} seconds")
        print("-" * 95)
        save_and_plot_results(results_summary, opt.model_name, modelPath)
    
    print("\n✅ Multi-confidence evaluation finished!")
    # ===================================================================