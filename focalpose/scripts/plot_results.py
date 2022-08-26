from matplotlib import pyplot as plt
import yaml
import argparse
from focalpose.config import RESULTS_DIR
import os
import numpy as np
from math import ceil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', default=None, help='Grouped directories with results to compare (comma-separated)')
    parser.add_argument('--labels', default=None, help='Comma-separated dir labels')
    parser.add_argument('--datasets', default=None, help='Comma-separated dataset names')

    pix3d_ds_names = ['pix3d-bed', 'pix3d-chair', 'pix3d-sofa', 'pix3d-table']
    all_ds_names = ['compcars3d', 'stanfordcars3d'] + pix3d_ds_names
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    args = parser.parse_args([] if '__file__' not in globals() else None)
    dirs = list(filter(None, args.dirs.split(',')))
    ds_names = all_ds_names if args.datasets is None else list(filter(None, args.datasets.split(',')))
    n_datasets = len(ds_names)
    labels = dirs  if  args.labels is None  else  list(filter(None, args.labels.split(',')))
    if len(labels) != len(dirs):
        raise Exception(f"Number of dirs {(len(dirs))} and labels ({len(labels)}) does not match.")


    # Load all results
    results = dict()
    pix3d_mean = ('pix3d' in ds_names)
    for dir in dirs:
        results[dir] = dict()

        for ds in set(ds_names + (pix3d_ds_names if pix3d_mean else [])) - set(['pix3d']):
            ds_dir = next((subdir for subdir in os.listdir(RESULTS_DIR / dir) if ds in subdir), None)
            if ds_dir is None:
                raise Exception(f"'{dir}' does not contain directory for '{ds}'.")
            with open(RESULTS_DIR / dir / ds_dir / 'results.yaml') as f:
                results[dir][ds] = yaml.safe_load(f)

    if 'pix3d' in ds_names:
        keys = results[dir][ds].keys()
        for dir in dirs:
            results[dir]['pix3d'] = dict()
            for key in keys:
                results[dir]['pix3d'][key] = np.mean([results[dir][ds][key] for ds in pix3d_ds_names])
                

    #keys = list(results[dir][ds].keys())
    #keys = ['R_acc_15_deg', 'R_acc_30_deg', 'R_acc_5_deg', 'R_err_median', 'Rt_err_median', 'acc_0.5', 'f_err_median', 'proj_acc_0.01', 'proj_acc_0.05', 'proj_acc_0.1', 'proj_err_median','ret_acc', 't_err_median']
    keys = ['proj_acc_0.1', 'R_acc_30_deg', 'Rt_err_median', 'f_err_median', 't_err_median']

    # Plotting
    width = 0.5
    for key in keys:
        if n_datasets <= 3:
            n_rows,n_cols = 1,n_datasets
        elif n_datasets == 4:
            n_rows,n_cols = 2,2
        elif n_datasets <= 6:
            n_rows,n_cols = 2,3
        else:
            n_rows,n_cols = ceil(n_datasets/4),4
        
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols,5*n_rows))
        if n_datasets == 1:
            axs = np.array([axs])
        else:
            axs = axs.reshape(-1)

        ymin = float('inf')
        ymax = float('-inf')
        for i in range(n_datasets):
            ax = axs[i]
            ds = ds_names[i]

            ys = [results[dir][ds][key] for dir in dirs]
            bars = ax.bar(labels, ys, color=colors)
            ymin = min(np.min(ys), ymin)
            ymax = max(np.max(ys), ymax)

            ax.set_title(ds_names[i])
            #ax.legend(ncol=3)
            #ax.get_xaxis().set_visible(False)

        for ax in axs:
            ax.set_ylim(bottom=ymin-(ymax-ymin)/8, top=ymax+(ymax-ymin)/8)

        fig.suptitle(key)
    plt.show()