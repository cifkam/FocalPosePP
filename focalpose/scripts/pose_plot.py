#!/usr/bin/env python
from sys import exit
import argparse
import numpy as np
import numpy.linalg
import numpy.random as nr
from pathlib import Path
from matplotlib import projections, pyplot as plt, patches
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation

from deep_bingham.bingham_distribution import BinghamDistribution
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset', default=None, help='{stanfordcars, compcars, pix3d-bed, pix3d-chair, pix3d-sofa, pix3d-table}')

parser.add_argument('--outliers', type=float, default=0.05,  help='Portion of points to remove from dataset as outliers.')
parser.add_argument('--overlay',  action='store_true', default=False, help='Overlay plots of real and fitted datasets.')

parser.add_argument('--train',    action='store_true', default=False, help='Use test dataset.')
parser.add_argument('--test',     action='store_true', default=False, help='Use test dataset.')
parser.add_argument('--fit',      action='store_true', default=False, help='Plot samples from fitted distributions.')

parser.add_argument('--cam', action='store_true', default=False, help='Plot camera positions.')
parser.add_argument('--t',   action='store_true', default=False, help='Plot translations vectors.')
parser.add_argument('--rx',  action='store_true', default=False, help='Plot rotated unit x vectors.')
parser.add_argument('--ry',  action='store_true', default=False, help='Plot rotated unit y vectors.')
parser.add_argument('--rz',  action='store_true', default=False, help='Plot rotated unit z vectors.')
parser.add_argument('--xy',  action='store_true', default=False, help='Plot x and y components of translation vectors.')
parser.add_argument('--zf',  action='store_true', default=False, help='Plot graph focal lengths and z components of translation vectors.')

parser.add_argument('--alpha', type=float, default=0.4, help="The alpha blending value of plot objects.")


pix3d_categories = ['bed', 'chair', 'sofa', 'table']
FIGSIZE_2D=(8,6)
FIGSIZE_3D=(7,6)
LEGEND_LOC='upper right'


def process(dataset, outliers, fit=True):
    if outliers > 0:
        t = dataset.TCO[:,:3,3]
        zf = np.vstack([t[:,2], dataset.f]).T
        dataset.index = dataset.index.drop(get_outliers(zf, outliers))

    d = dict()
    mat = dataset.TCO
    R = mat[:,:3,:3]
    t = mat[:,:3, 3]
    f = dataset.f
    cam_poses = (-R@t[:, :, None]).squeeze()

    xy = t[:,:2]
    xy_mu = np.mean(xy, axis=0)
    xy_cov = np.cov(xy.T)

    d['R']         = R
    d['t']         = t
    d['f']         = f
    d['cam_pos']   = cam_poses

    if fit:
        R_quat = np.array(list(  map(lambda x: Rotation.from_matrix(x).as_quat(), R)  ))
        bingham = BinghamDistribution.fit(R_quat)
        zf = np.vstack([t[:,2],f]).T
        logzf = np.log(zf)
        zf_log_mu = np.mean(logzf, axis=0)
        zf_log_cov = np.cov(logzf.T)

        d['bingham_z'] = bingham._param_z
        d['bingham_m'] = bingham._param_m
        d['xy_mu']     = xy_mu
        d['xy_cov']    = xy_cov
        d['zf_log_mu']  = zf_log_mu
        d['zf_log_cov'] = zf_log_cov

    return d


def set_xyz_labels(ax, z=True):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if z: ax.set_zlabel('z')

def plot_cam_pos(d, ds_name, label, overlay, sample=False, ax=None):
    if ax is None:  
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = plt.axes(projection='3d')
        set_xyz_labels(ax)
        fig.suptitle('Cam positions: ' + ds_name)
        ax.scatter([0], [0], [0], c='r')

    pos = d['cam_pos']

    if sample:
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        bingham   = BinghamDistribution(d['bingham_m'], d['bingham_z'])
        zf_log_mu  = d['zf_log_mu']
        zf_log_cov = d['zf_log_cov']

        R = np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), bingham.random_samples(pos.shape[0]))))
        xy = nr.multivariate_normal(xy_mu, xy_cov, size=pos.shape[0])
        z = np.exp(nr.multivariate_normal(zf_log_mu, zf_log_cov, size=pos.shape[0]))[:,0]
        t = np.hstack([xy,z.reshape(-1,1)])
        samples = (-R@t[:, :, None]).squeeze()
        ax.scatter(samples[:,0], samples[:,1], samples[:,2], label=label, alpha=args.alpha)
    else:
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], label=label, alpha=args.alpha)

    if not overlay:
        ax.legend(loc=LEGEND_LOC)

    return ax





def plot_trans(trans, ds_name):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = plt.axes(projection='3d')
    set_xyz_labels(ax)

    ax.title.set_text('Translations: ' + ds_name)
    ax.scatter(trans[:,0], trans[:,1], trans[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')




def plot_rot_axis(d, axis, ds_name, label, overlay, sample=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = plt.axes(projection='3d')
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        set_xyz_labels(ax)
        ax.title.set_text('Rotations ('+ ('x' if axis==0 else 'y' if axis==1 else 'z') + '-axis): ' + ds_name)

    rot = d['R']
    unit = np.array([0,0,0])
    unit[axis] = 1

    if sample:
        bingham   = BinghamDistribution(d['bingham_m'], d['bingham_z'])
        sample_rot = np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), bingham.random_samples(rot.shape[0]))))
        pts = sample_rot @ unit
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=args.alpha, label=label)
    else:
        pts = rot @ unit
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=args.alpha, label=label)

    if not overlay:
        ax.legend(loc=LEGEND_LOC)

    return ax


def plot_xy(d, ds_name, label, overlay, sample=False, ax=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=FIGSIZE_2D)
        set_xyz_labels(ax,z=False)
        ax.set_title('x:y : ' + ds_name)
        ax.add_patch(patches.Rectangle((-0.15, -0.15), 0.3, 0.3, alpha=args.alpha, color='k', label='FocalPose synt. data'))

    xy = d['t'][:,:2]
    if sample:
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        samples = nr.multivariate_normal(xy_mu, xy_cov, size=xy.shape[0])
        ax.scatter(samples[:,0], samples[:,1], alpha=args.alpha, label=label)
    else:
        ax.scatter(xy[:,0], xy[:,1], alpha=args.alpha, label=label)

    if not overlay:
        ax.legend(loc=LEGEND_LOC)

    return ax


def plot_zf(d, ds_name, label, overlay, sample=False, ax=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=FIGSIZE_2D)
        ax.set_xlabel('z')
        ax.set_ylabel('f')
        ax.set_title('z:f : ' + ds_name)

        if ds_name == 'pix3d-chair':  z_interval = (0.8, 3.4)  
        if ds_name == 'stanfordcars': z_interval = (0.8, 3.0)
        if ds_name == 'compcars':     z_interval = (0.8, 3.0)
        else:                         z_interval = (0.8, 2.4)
        ax.add_patch(patches.Rectangle((z_interval[0], 200), z_interval[1]-z_interval[0], 800, alpha=args.alpha, color='k', label='FocalPose synt. data'))

    z = d['t'][:,2]
    f = d['f']

    if sample:
        zf_log_mu  = d['zf_log_mu']
        zf_log_cov = d['zf_log_cov']
        samples = np.exp(nr.multivariate_normal(zf_log_mu, zf_log_cov, size=z.shape[0]))
        ax.scatter(samples[:,0], samples[:,1], alpha=args.alpha, label=label)
    else:
        ax.scatter(z, f, alpha=args.alpha, label=label)

    if not overlay:
        ax.legend(loc=LEGEND_LOC)

    return ax



def plot(args, ds_name, dict_train, dict_test):
    for plot, plot_funct in [(args.zf, plot_zf), (args.xy, plot_xy), (args.cam, plot_cam_pos)]:
        if plot:
            if args.train:      ax = plot_funct(dict_train, ds_name, 'train',      args.overlay)
            if args.fit:        ax = plot_funct(dict_train, ds_name, 'parametric', args.overlay, ax=ax, sample=True)
            if args.test:       ax = plot_funct(dict_test,  ds_name, 'test',       False,        ax=ax)

    for axis, plot_rot in enumerate([args.rx, args.ry, args.rz]):
        if plot_rot:
            if args.train:  ax = plot_rot_axis(dict_train, axis, ds_name, 'train',       args.overlay)
            if args.fit:    ax = plot_rot_axis(dict_train, axis, ds_name, 'parameteric', args.overlay, ax=ax, sample=True)
            if args.test:   ax = plot_rot_axis(dict_test,  axis, ds_name, 'test',        False,        ax=ax)

    if args.t:   plot_trans(dict_train['t'], ds_name)
    plt.show()


def get_outliers(data, q=0.05):
    med = np.median(data, axis=0)
    dist = np.sqrt(np.sum((data - med)**2, axis=-1))
    n = int(data.shape[0]*q)
    return np.argpartition(-dist, n)[:n]                                          




if __name__ == '__main__':
    args = parser.parse_args([] if '__file__' not in globals() else None)

    dict_test = None
    if args.dataset[:5] == 'pix3d':
        c = args.dataset[6:]
        categories = pix3d_categories if c == '' else c.split(',') 

        for category in categories:
            ds_train = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category, train=True)
            dict_train = process(ds_train, args.outliers)

            if args.test:
                ds_test = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category, train=False)
                dict_test = process(ds_test, 0, fit=False)
            plot(args, 'pix3d-'+category, dict_train, dict_test)

    elif args.dataset == 'compcars':
        ds_train = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', train=True)
        dict_train = process(ds_train, args.outliers)

        if args.test:
                ds_test = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', train=False)
                dict_test = process(ds_test, 0, fit=False)
        plot(args, 'CompCars', dict_train, dict_test)

    elif args.dataset == 'stanfordcars':
        ds_train = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', train=True)
        dict_train = process(ds_train, args.outliers)

        if args.test:
                ds_test = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', train=False)
                dict_test = process(ds_test, 0, fit=False)
        plot(args, 'StanfordCars', dict_train, dict_test)

    else:
        parser.print_help()
        exit(1)
