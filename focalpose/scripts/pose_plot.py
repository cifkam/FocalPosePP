#!/usr/bin/env python
from sys import exit
import argparse
import numpy as np
import numpy.linalg
import numpy.random as nr
import pandas as pd
from pathlib import Path
from matplotlib import projections, pyplot as plt, patches
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation

from deep_bingham.bingham_distribution import BinghamDistribution
from focalpose.config import LOCAL_DATA_DIR, SYNT_DS_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset

parser = argparse.ArgumentParser()
parser.add_argument('dataset', default=None, help='{stanfordcars, compcars, pix3d-bed, pix3d-chair, pix3d-sofa, pix3d-table}')

parser.add_argument('--outliers', type=float,          default=0.05,  help='Portion of points to remove from dataset as outliers.')
parser.add_argument('--separate',  action='store_true', default=False, help='Do not overlay plots.')

parser.add_argument('--train',    action='store_true', default=False, help='Use test dataset.')
parser.add_argument('--test',     action='store_true', default=False, help='Use test dataset.')
parser.add_argument('--fit',      action='store_true', default=False, help='Plot samples from fitted distributions.')
parser.add_argument('--synts',    type=str,            default='',  help='Comma-separated synt dataset dirs to plot.')

parser.add_argument('--cam', action='store_true', default=False, help='Plot camera positions.')
parser.add_argument('--t',   action='store_true', default=False, help='Plot translations vectors.')
parser.add_argument('--rx',  action='store_true', default=False, help='Plot rotated unit x vectors.')
parser.add_argument('--ry',  action='store_true', default=False, help='Plot rotated unit y vectors.')
parser.add_argument('--rz',  action='store_true', default=False, help='Plot rotated unit z vectors.')
parser.add_argument('--xy',  action='store_true', default=False, help='Plot x and y components of translation vectors.')
parser.add_argument('--zf',  action='store_true', default=False, help='Plot graph focal lengths and z components of translation vectors.')

parser.add_argument('--alpha', type=float, default=0.36, help="The alpha blending value of plot objects.")


pix3d_categories = ['bed', 'chair', 'sofa', 'table']
FIGSIZE_2D=(8,6)
FIGSIZE_3D=(7,6)
LEGEND_LOC='upper right'


def process_synts(dirs, n_samples):
    synts = []
    for dir in dirs:
        df = pd.read_pickle(SYNT_DS_DIR / dir / 'camera.pkl')
        df = df.sample(n_samples*3)
        data = dict()
        data['TCO'] = np.array(df['TCO'].to_list())
        data['f'] = np.array(df['K'].to_list())[:,0,0]
        synts.append((dir,process(data, fit=False)))
    return synts
    
def process_real(dataset, outliers=0.0, fit=True):
    if outliers > 0:
        t = dataset.TCO[:,:3,3]
        zf = np.vstack([t[:,2], dataset.f]).T
        dataset.index = dataset.index.drop(get_outliers(zf, outliers))

    data = dict()
    data['TCO'] = dataset.TCO
    data['f'] = dataset.f
    return process(data, fit)

def process(data, fit=True):
    mat = data['TCO']
    R = mat[:,:3,:3]
    t = mat[:,:3, 3]
    f = data['f']
    cam_poses = (-R@t[:, :, None]).squeeze()

    xy = t[:,:2]
    xy_mu = np.mean(xy, axis=0)
    xy_cov = np.cov(xy.T)

    processed = dict()
    processed['R']         = R
    processed['t']         = t
    processed['f']         = f
    processed['cam_pos']   = cam_poses

    if fit:
        R_quat = np.array(list(  map(lambda x: Rotation.from_matrix(x).as_quat(), R)  ))
        bingham = BinghamDistribution.fit(R_quat)
        zf = np.vstack([t[:,2],f]).T
        logzf = np.log(zf)
        zf_log_mu = np.mean(logzf, axis=0)
        zf_log_cov = np.cov(logzf.T)

        processed['bingham_z'] = bingham._param_z
        processed['bingham_m'] = bingham._param_m
        processed['xy_mu']     = xy_mu
        processed['xy_cov']    = xy_cov
        processed['zf_log_mu']  = zf_log_mu
        processed['zf_log_cov'] = zf_log_cov

    return processed


def set_xyz_labels(ax, z=True):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if z: ax.set_zlabel('z')

def plot_cam(d, ds_name, label, separate, sample=False, ax=None):
    if ax is None or separate:
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = plt.axes(projection='3d')
        set_xyz_labels(ax)
        fig.suptitle('Cam positions: ' + ds_name)
        ax.scatter([0], [0], [0], c='k')

    pos = d['cam_pos']

    if sample:
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        zf_log_mu  = d['zf_log_mu']
        zf_log_cov = d['zf_log_cov']
        bingham   = BinghamDistribution(d['bingham_m'], d['bingham_z'])

        n_samples = pos.shape[0]
        R = np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), bingham.random_samples(n_samples))))
        xy = nr.multivariate_normal(xy_mu, xy_cov, size=n_samples)
        z = np.exp(nr.multivariate_normal(zf_log_mu, zf_log_cov, size=n_samples))[:,0]
        t = np.hstack([xy,z.reshape(-1,1)])
        pos = (-R@t[:, :, None]).squeeze()
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], label=label, alpha=args.alpha)

    if separate:
        ax.legend(loc=LEGEND_LOC)

    return ax


def plot_trans(d, ds_name, label, separate, sample=False, ax=None):
    if ax is None or separate:
        fig = plt.figure(figsize=FIGSIZE_3D)
        ax = plt.axes(projection='3d')
        set_xyz_labels(ax)
        fig.suptitle('Translations: ' + ds_name)
        ax.scatter([0], [0], [0], c='k')

    trans = d['t']

    if sample:
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        zf_log_mu  = d['zf_log_mu']
        zf_log_cov = d['zf_log_cov']

        n_samples = trans.shape[0]
        xy = nr.multivariate_normal(xy_mu, xy_cov, size=n_samples)
        z = np.exp(nr.multivariate_normal(zf_log_mu, zf_log_cov, size=n_samples))[:,0]
        trans = np.hstack([xy,z.reshape(-1,1)])
    ax.scatter(trans[:,0], trans[:,1], trans[:,2], label=label, alpha=args.alpha)

    if separate:
        ax.legend(loc=LEGEND_LOC)

    return ax


def plot_rot_axis(d, axis, ds_name, label, separate, sample=False, ax=None):
    if ax is None or separate:
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
    pts = rot @ unit

    if sample:
        bingham   = BinghamDistribution(d['bingham_m'], d['bingham_z'])
        n_samples = rot.shape[0]
        sample_rot = np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), bingham.random_samples(n_samples))))
        pts = sample_rot @ unit
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=args.alpha, label=label)

    if separate:
        ax.legend(loc=LEGEND_LOC)

    return ax

def plot_rot_x(d, ds_name, label, separate, sample=False, ax=None):
    return plot_rot_axis(d, 0, ds_name, label, separate, sample, ax)
def plot_rot_y(d, ds_name, label, separate, sample=False, ax=None):
    return plot_rot_axis(d, 1, ds_name, label, separate, sample, ax)
def plot_rot_z(d, ds_name, label, separate, sample=False, ax=None):
    return plot_rot_axis(d, 2, ds_name, label, separate, sample, ax)

def plot_xy(d, ds_name, label, separate, sample=False, ax=None):
    if ax is None or separate:
        fig,ax = plt.subplots(figsize=FIGSIZE_2D)
        set_xyz_labels(ax,z=False)
        ax.set_title('x:y : ' + ds_name)
        #ax.add_patch(patches.Rectangle((-0.15, -0.15), 0.3, 0.3, alpha=args.alpha, color='k', label='FocalPose synt. data'))

    xy = d['t'][:,:2]
    if sample:
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        n_samples = xy.shape[0]
        xy = nr.multivariate_normal(xy_mu, xy_cov, size=n_samples)
    ax.scatter(xy[:,0], xy[:,1], alpha=args.alpha, label=label)

    if separate:
        ax.legend(loc=LEGEND_LOC)

    return ax


def plot_zf(d, ds_name, label, separate, sample=False, ax=None):
    if ax is None or separate:
        fig,ax = plt.subplots(figsize=FIGSIZE_2D)
        ax.set_xlabel('z')
        ax.set_ylabel('f')
        ax.set_title('z:f : ' + ds_name)

        #if ds_name == 'pix3d-chair':  z_interval = (0.8, 3.4)  
        #if ds_name == 'stanfordcars': z_interval = (0.8, 3.0)
        #if ds_name == 'compcars':     z_interval = (0.8, 3.0)
        #else:                         z_interval = (0.8, 2.4)
        #ax.add_patch(patches.Rectangle((z_interval[0], 200), z_interval[1]-z_interval[0], 800, alpha=args.alpha, color='k', label='FocalPose synt. data'))

    z = d['t'][:,2]
    f = d['f']

    if sample:
        zf_log_mu  = d['zf_log_mu']
        zf_log_cov = d['zf_log_cov']
        n_samples = z.shape[0]
        samples = np.exp(nr.multivariate_normal(zf_log_mu, zf_log_cov, size=n_samples))
        ax.scatter(samples[:,0], samples[:,1], alpha=args.alpha, label=label)
    else:
        ax.scatter(z, f, alpha=args.alpha, label=label)

    if separate:
        ax.legend(loc=LEGEND_LOC)

    return ax



def plot(args, ds_name, dict_train, dict_test, synts):
    for plot, plot_funct in [ (args.zf, plot_zf), (args.xy, plot_xy), (args.cam, plot_cam), (args.t, plot_trans), (args.rx, plot_rot_x), (args.ry, plot_rot_y), (args.rz, plot_rot_z) ]:
        if plot:
            ax = None
            if args.train:      ax = plot_funct(dict_train, ds_name, 'train',      args.separate, ax=ax)
            if args.test:       ax = plot_funct(dict_test,  ds_name, 'test',       args.separate, ax=ax)
            if args.fit:        ax = plot_funct(dict_train, ds_name, 'parametric', args.separate, ax=ax, sample=True)

            for label,dict_synt in synts:
                ax = plot_funct(dict_synt, ds_name, label+' [synt]', args.separate, ax=ax)

            if not args.separate:
                ax.legend(loc=LEGEND_LOC)

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
        category = args.dataset[6:]
        ds_train = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category, train=True)
        dict_train = process_real(ds_train, args.outliers)
        synts = process_synts(filter(None,args.synts.split(',')), n_samples=dict_train['R'].shape[0])

        if args.test:
            ds_test = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category, train=False)
            dict_test = process_real(ds_test, 0, fit=False)
        plot(args, 'pix3d-'+category, dict_train, dict_test, synts)

    elif args.dataset == 'compcars':
        ds_train = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', train=True)
        dict_train = process_real(ds_train, args.outliers)
        synts = process_synts(filter(None,args.synts.split(',')), n_samples=dict_train['R'].shape[0])

        if args.test:
                ds_test = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', train=False)
                dict_test = process_real(ds_test, 0, fit=False)
        plot(args, 'CompCars', dict_train, dict_test, synts)

    elif args.dataset == 'stanfordcars':
        ds_train = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', train=True)
        dict_train = process_real(ds_train, args.outliers)
        synts = process_synts(filter(None,args.synts.split(',')), n_samples=dict_train['R'].shape[0])

        if args.test:
                ds_test = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', train=False)
                dict_test = process_real(ds_test, 0, fit=False)
        plot(args, 'StanfordCars', dict_train, dict_test, synts)

    else:
        parser.print_help()
        exit(1)
