import os
from collections import defaultdict
from functools import partial
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import sewar


def read_img(path):
    img = cv2.imread(path, -1)
    return img


def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_img_similarity(files, binary_metrics):
    files_count = len(files)
    metrics = defaultdict(partial(np.empty, (files_count, files_count)))

    for left_id in range(files_count):
        for right_id in range(left_id, files_count):
            if left_id == right_id:
                continue

            first_img = read_img(files[left_id])
            second_img = read_img(files[right_id])

            for metric_name, binary_metric in binary_metrics.items():
                print(f'Running {metric_name} for {files[left_id]} and {files[right_id]}')
                metric = binary_metric(first_img, second_img)
                metrics[metric_name][left_id][right_id] = metric
                metrics[metric_name][right_id][left_id] = metric

    return metrics


def compute_compute_similarity_in_dir(directory_path, binary_metrics):
    matrix_by_name = dict()

    for directory in directory_path:
        files = [os.path.join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
        results = compute_img_similarity(files, binary_metrics)
        matrix_by_name[os.path.basename(os.path.normpath(directory))] = results

    return matrix_by_name


if __name__ == '__main__':
    dirs = [r'.\pics\corrected', r'.\pics\orig']

    metrics = dict(mse=sewar.full_ref.mse,
                   rmse=sewar.full_ref.rmse,
                   psnr=sewar.full_ref.psnr,
                   # rmse_sw=sewar.full_ref.rmse_sw,
                   uqi=sewar.full_ref.uqi,
                   # ssim=sewar.full_ref.ssim,
                   ergas=sewar.full_ref.ergas,
                   scc=sewar.full_ref.scc,
                   rase=sewar.full_ref.rase,
                   sam=sewar.full_ref.sam,
                   msssim=sewar.full_ref.msssim,
                   vifp=sewar.full_ref.vifp,
                   psnrb=sewar.full_ref.psnrb,
                   )

    matrix_by_name = compute_compute_similarity_in_dir(dirs, metrics)

    print(matrix_by_name)
    # img_scaled = cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

    # ssim, diff = structural_similarity(first_img, second_img, full=True, channel_axis=2)
    # print(ssim)

    # cv2.imshow("Display window", img_scaled)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
