#!/usr/bin/env python
from sys import exit
import argparse
import numpy as np
import numpy.linalg
import numpy.random as nr
from pathlib import Path
from scipy.spatial.transform import Rotation
import json

from deep_bingham.bingham_distribution import BinghamDistribution
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset

parser = argparse.ArgumentParser()
parser.add_argument('--outliers', type=float, default=0.05,  help='Portion of points to remove from dataset as outliers.')

def fit(dataset, outliers):
    if outliers > 0:
        d = fit(dataset, 0)
        t = dataset.TCO[:,:3,3]
        zf = np.vstack([t[:,2], dataset.f]).T
        dataset.index = dataset.index.drop(get_outliers(zf, outliers))

    d = dict()
    mat = dataset.TCO
    R = mat[:,:3,:3]
    t = mat[:,:3,3]
    f = dataset.f

    R_quat = np.array(list(  map(lambda x: Rotation.from_matrix(x).as_quat(), R)  ))
    bingham = BinghamDistribution.fit(R_quat)

    xy = t[:,:2]
    xy_mu = np.mean(xy, axis=0)
    xy_cov = np.cov(xy.T)

    zf = np.vstack([t[:,2],f]).T
    logzf = np.log(zf)
    zf_log_mu = np.mean(logzf, axis=0)
    zf_log_cov = np.cov(logzf.T)

    d['rot_bingham_z'] = bingham._param_z.tolist()
    d['rot_bingham_m'] = bingham._param_m.tolist()

    d['xy_mu']     = xy_mu.tolist()
    d['xy_cov']    = xy_cov.tolist()

    d['zf_log_mu']  = zf_log_mu.tolist()
    d['zf_log_cov'] = zf_log_cov.tolist()

    return d


def get_outliers(data, q=0.05):
    med = np.median(data, axis=0)
    dist = np.sqrt(np.sum((data - med)**2, axis=-1))
    n = int(data.shape[0]*q)
    return np.argpartition(-dist, n)[:n]                                          


if __name__ == '__main__':
    args = parser.parse_args([] if '__file__' not in globals() else None)

    # Pix3D
    pix3d_categories = ['bed', 'chair', 'sofa', 'table']
    for category in pix3d_categories:
        dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category)
        d = fit(dataset, args.outliers)
        with open(LOCAL_DATA_DIR / 'pix3d' / f"{category}-fit.json", 'w') as f:
            f.write(json.dumps(d))

    # CompCars
    dataset = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars')
    d = fit(dataset, args.outliers)
    with open(LOCAL_DATA_DIR / 'CompCars' / "fit.json", 'w') as f:
        f.write(json.dumps(d))

    # StanfordCars
    dataset = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars')
    d = fit(dataset, args.outliers)
    with open(LOCAL_DATA_DIR / 'StanfordCars' / 'fit.json', 'w') as f:
        f.write(json.dumps(d))
        