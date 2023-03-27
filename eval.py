from pathlib import Path
import numpy as np
import cv2


def mae(gt, pred):
    return np.sum(np.abs(gt - pred)) / gt.size


def mse(gt, pred):
    error = gt - pred
    return np.sum(error * error) / gt.size


def divide_gt(gt: np.ndarray, x_center, y_center):
    gt_tl = gt[:y_center, :x_center]
    gt_tr = gt[:y_center, x_center:]
    gt_bl = gt[y_center:, :x_center]
    gt_br = gt[y_center:, x_center:]

    # other implementations are sussy
    fg_pixels = np.sum(gt)

    w1 = np.sum(gt_tl) / fg_pixels
    w2 = np.sum(gt_tr) / fg_pixels
    w3 = np.sum(gt_bl) / fg_pixels
    w4 = np.sum(gt_br) / fg_pixels


    return gt_tl, gt_tr, gt_bl, gt_br, w1, w2, w3, w4


def divide_pred(pred: np.ndarray, x_center, y_center):
    pred_tl = pred[:y_center, :x_center]
    pred_tr = pred[:y_center, x_center:]
    pred_bl = pred[y_center:, :x_center]
    pred_br = pred[y_center:, x_center:]

    return pred_tl, pred_tr, pred_bl, pred_br


def ssim(gt: np.ndarray, pred: np.ndarray):
    x = gt.flatten()
    y = pred.flatten()

    mean_x = np.mean(gt)
    mean_y = np.mean(pred)
    sig_x = np.std(gt)
    sig_y = np.std(pred)

    cov_xy = np.sum((x - mean_x) * (y - mean_y)) / x.shape[0]

    ssim_1 = (2 * mean_x * mean_y) / (mean_x ** 2 + mean_y ** 2 + 1e-20)
    ssim_2 = (2 * sig_x * sig_y) / (sig_x ** 2 + sig_y ** 2 + 1e-20)
    ssim_3 = cov_xy / (sig_x * sig_y + 1e-20)
    return ssim_1 * ssim_2 * ssim_3


def s_region(gt: np.ndarray, pred: np.ndarray):
    # x_center, y_center = centroid(gt)
    x_center, y_center = gt.shape[1] // 2, gt.shape[0] // 2

    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = divide_gt(gt, x_center, y_center)
    pred_1, pred_2, pred_3, pred_4 = divide_pred(pred, x_center, y_center)

    ssim_1 = ssim(gt_1, pred_1)
    ssim_2 = ssim(gt_2, pred_2)
    ssim_3 = ssim(gt_3, pred_3)
    ssim_4 = ssim(gt_4, pred_4)

    return ssim_1 * w1 + ssim_2 * w2 + ssim_3 * w3 + ssim_4 * w4


def s_object(gt: np.ndarray, pred: np.ndarray):
    LAMBDA = 0.5
    pred_fg = pred * gt

    x_mean_fg = np.mean(pred_fg[gt == 1])
    x_sig_fg = np.std(pred_fg[gt == 1])

    o_fg = (2 * x_mean_fg) / (x_mean_fg ** 2 + 1 + 2 * LAMBDA * x_sig_fg)

    pred_bg = (1.0 - pred) * (1.0 - gt)
    x_mean_bg = np.mean(pred_bg[gt == 0])
    x_sig_bg = np.std(pred_bg[gt == 0])

    o_bg = (2 * x_mean_bg) / (x_mean_bg ** 2 + 1 + 2 * LAMBDA * x_sig_bg)

    mu = np.sum(gt) / (gt.shape[0] * gt.shape[1])

    return mu * o_fg + (1 - mu) * o_bg


def s_measure(gt, pred):
    ALPHA = 0.5

    y = np.mean(gt)
    if y == 0:
        return 1 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)
    else:
        return ALPHA * s_object(gt, pred) + (1 - ALPHA) * s_region(gt, pred)


def main():
    gts = sorted(Path("/home/matthias/Downloads/COD10K-v3/Test/GT_Object").iterdir())
    preds = sorted(Path("/home/matthias/Downloads/COD10K-v3-Hitnet").iterdir())

    mae_sum = 0
    s_measure_sum = 0

    for gt_file, pred_file in zip(gts, preds):
        gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)

        gt = gt / 255
        pred = pred / 255

        mae_sum += mae(gt, pred)
        s_measure_sum += s_measure(gt, pred)

    print(f"MAE: {mae_sum / len(gts)}")
    print(f"S Measure: {s_measure_sum / len(gts)}")



if __name__ == "__main__":
    gt = cv2.imread("/home/matthias/Downloads/COD10K-v3/Test/GT_Object/COD10K-CAM-1-Aquatic-3-Crab-23.png", cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread("/home/matthias/Downloads/COD10K-v3-Hitnet/COD10K-CAM-1-Aquatic-3-Crab-23.png", cv2.IMREAD_GRAYSCALE)

    # gt = gt / 255
    # pred = pred / 255

    # print(s_measure(gt, pred))

    main()
