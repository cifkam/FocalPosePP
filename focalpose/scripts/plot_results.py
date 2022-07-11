from matplotlib import pyplot as plt
import yaml
import argparse
from focalpose.config import RESULTS_DIR
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dirs', default=None, help='Grouped directories with results to compare (comma separated)')

ds_names = ['compcars3d', 'stanfordcars3d', 'pix3d-bed', 'pix3d-chair', 'pix3d-sofa', 'pix3d-table']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


if __name__ == '__main__':
    args = parser.parse_args([] if '__file__' not in globals() else None)
    dirs = args.dirs.split(',')

    # Load all results
    results = dict()
    for dir in dirs:
        results[dir] = dict()

        for ds in ds_names:
            ds_dir = next((subdir for subdir in os.listdir(RESULTS_DIR / dir) if ds in subdir), None)
            with open(RESULTS_DIR / dir / ds_dir / 'results.yaml') as f:
                results[dir][ds] = yaml.safe_load(f)

    #keys = list(results[dir][ds].keys())
    #keys = ['R_acc_15_deg', 'R_acc_30_deg', 'R_acc_5_deg', 'R_err_median', 'Rt_err_median', 'acc_0.5', 'f_err_median', 'proj_acc_0.01', 'proj_acc_0.05', 'proj_acc_0.1', 'proj_err_median','ret_acc', 't_err_median']
    keys = ['proj_acc_0.1', 'R_acc_30_deg', 'Rt_err_median', 'f_err_median', 't_err_median']

    # Plotting
    width = 0.5
    for key in keys:
        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15,10))
        axs = axs.reshape(-1)

        ymin = float('inf')
        ymax = float('-inf')
        for i in range(6):

            ax = axs[i]
            ds = ds_names[i]

            n = len(dirs)

            ys = [results[dir][ds][key] for dir in dirs]
            bars = ax.bar(dirs, ys, color=colors)
            ymin = min(np.min(ys), ymin)
            ymax = max(np.max(ys), ymax)

            ax.set_title(ds_names[i])
            #ax.legend(ncol=3)
            #ax.get_xaxis().set_visible(False)

        for ax in axs:
            ax.set_ylim(bottom=ymin-(ymax-ymin)/8, top=ymax+(ymax-ymin)/8)

        fig.suptitle(key)
    plt.show()

