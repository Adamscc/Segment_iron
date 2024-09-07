import os
import torch
import numpy as np


def count_model_parameters(model_path):
    """
    计算模型参数总量
    :param model_path: 模型文件路径（.pt或.pth），要求选手使用 torch.save(model, 'model.pth') 保存模型
    :return: 模型参数总量
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    # 加载模型
    model = torch.load(model_path)

    # 计算参数总量
    total_params = sum(p.numel() for p in model.parameters())

    return total_params


def calculate_iou(pred, gt, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def seg(pred_dir, gt_dir, num_classes):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

    assert len(pred_files) == len(gt_files), "Prediction and GT files count do not match"

    all_ious = np.zeros((len(pred_files), num_classes))

    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        pred = np.load(os.path.join(pred_dir, pred_file))
        gt = np.load(os.path.join(gt_dir, gt_file))

        ious = calculate_iou(pred, gt, num_classes)
        all_ious[i] = ious

    mean_ious = np.mean(all_ious, axis=0)

    return mean_ious


if __name__ == "__main__":
    # 训练好的模型的路径
    model_path = r'D:\AILAB\校园算法\挑战赛\base_unet\checkpoints\Unet3p_20_train2.pt'

    score = 0

    ### 计算模型参数分数 ###
    total_params = count_model_parameters(model_path)
    norm_params = total_params / 1_000_000
    print(f"模型的参数总量为: {norm_params:.2f} M.")
    score_para = 0
    if norm_params > 17:
        score_para = 10
    else:
        if norm_params < 1:
            score_para = 70
        else:
            score_para = 70 - 15 / 4 * (norm_params - 1)
    print(f"模型参数的分数为: {score_para}")
    score += score_para
    ###################

    #### 计算 Class IoU 分数 ####
    pred_dir = 'test_predictions/'
    base_dir = 'baseline_predictions/'
    gt_dir = 'test_ground_truths/'
    num_classes = 3  # 异常类型数
    improvement_threshold = 0.06
    pre_IoU = seg(pred_dir, gt_dir, num_classes)
    base_IoU = seg(base_dir, gt_dir, num_classes)

    thr = 130
    for cls_index, (pre, base) in enumerate(zip(pre_IoU, base_IoU)):
        delta = pre - base
        print(f"Class {cls_index} \n预测 IoU: {pre}, \n基准 IoU: {base}, \n差值: {delta}")

        if delta >= improvement_threshold:
            score_class = 100
        else:
            if delta <= 0:
                score_class = 0
            else:
                score_class = 40 + (thr * delta) ** 2
        print(f"Class {cls_index} - 分数: {score_class}")
        score += score_class

    print(f"最终分数: {score}")
