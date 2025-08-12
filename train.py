from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data

# ==================== [核心修改 1: 导入正确的模块] ====================
from lib.utils.opts import opts  # 只导入 opts 类
# ====================================================================

from lib.utils.logger import Logger
from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.coco_rsdata import COCO
from lib.Trainer.ctdet import CtdetTrainer


def main(opt):
    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # --- 数据加载器设置 ---
    DataTrain = COCO(opt, 'train')
    train_loader = torch.utils.data.DataLoader(
        DataTrain, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )
    DataVal = COCO(opt, 'test')
    val_loader = torch.utils.data.DataLoader(
        DataVal, batch_size=1, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )

    print('Creating model...')
    head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    model = get_det_net(head, opt.model_name)
    print(f"Model: {opt.model_name}")

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    logger = Logger(opt)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')
    best = -1
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write(f'epoch: {epoch} |')

        # 使用 opt.save_weights_dir 保存模型
        save_model(os.path.join(opt.save_weights_dir, 'model_last.pth'), epoch, model, optimizer)

        for k, v in log_dict_train.items():
            logger.write(f'{k} {v:8f} | ')

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_weights_dir, f'model_{epoch}.pth'), epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, _, stats = trainer.val(epoch, val_loader, DataVal.coco, DataVal)
            for k, v in log_dict_val.items():
                logger.write(f'{k} {v:8f} | ')
            logger.write('eval results: ')
            for k in stats.tolist():
                logger.write(f'{k:8f} | ')
            if log_dict_val['ap50'] > best:
                best = log_dict_val['ap50']
                save_model(os.path.join(opt.save_weights_dir, 'model_best.pth'), epoch, model)

        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_weights_dir, f'model_{epoch}.pth'), epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    # ==================== [核心修改 2: 新的启动流程] ====================
    # 1. 创建 opts 对象
    opts_parser = opts()
    # 2. 解析命令行参数
    opt = opts_parser.parse()
    # 3. 初始化路径等派生变量
    opt = opts_parser.init(opt)
    # 4. 将配置好的 opt 对象传入 main 函数
    main(opt)
    # =================================================================