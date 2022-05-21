#!/usr/bin/env python

from sys import exit
import argparse
import numpy as np
import numpy.random as nr
from pathlib import Path
from matplotlib import projections, pyplot as plt, patches
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation

from deep_bingham.bingham_distribution import BinghamDistribution
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset
import numpy.linalg

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str,   default=None, help='{stanfordcars, compcars, pix3d-{bed, chair, sofa, table} }')
parser.add_argument('--outliers', type=float, default=0.0,  help='Portion of points to remove from dataset as outliers.')
parser.add_argument('--test',     action='store_true', default=False, help='Use test dataset.')
parser.add_argument('--fit',      action='store_true', default=False, help='Plot samples from fitted distributions.')
parser.add_argument('--overlay',  action='store_true', default=False, help='Overlay plots of real and fitted datasets.')

parser.add_argument('--cam', action='store_true', default=False, help='Plot camera positions.')
parser.add_argument('--t',   action='store_true', default=False, help='Plot translations vectors.')
parser.add_argument('--rx',  action='store_true', default=False, help='Plot rotated unit x vectors.')
parser.add_argument('--ry',  action='store_true', default=False, help='Plot rotated unit y vectors.')
parser.add_argument('--rz',  action='store_true', default=False, help='Plot rotated unit z vectors.')
parser.add_argument('--xy',  action='store_true', default=False, help='Plot x and y components of translation vectors.')
parser.add_argument('--zf',  action='store_true', default=False, help='Plot graph focal lengths and z components of translation vectors.')

pix3d_categories = ['bed', 'chair', 'sofa', 'table']
ALPHA = 0.25


def process(dataset, outliers):
    if outliers > 0:
        d = process(dataset, False)
        zf = np.vstack([d['t'][:,2], d['f']]).T
        dataset.index = dataset.index.drop(get_outliers(zf, outliers))

    d = dict()
    R = dataset.R
    t = dataset.t
    f = dataset.f
    cam_poses = (-R@t[:, :, None]).squeeze()

    R_quat = np.array(list(  map(lambda x: Rotation.from_matrix(x).as_quat(), R)  ))
    bingham = BinghamDistribution.fit(R_quat)

    xy = t[:,:2]
    xy_mu = np.mean(xy, axis=0)
    xy_cov = np.cov(xy.T)

    zf = np.vstack([t[:,2],f]).T
    logzf = np.log(zf)
    logzf_mu = np.mean(logzf, axis=0)
    logzf_cov = np.cov(logzf.T)

    d['R']         = R
    d['t']         = t
    d['cam_pos']   = cam_poses
    d['R_quat']    = R_quat
    d['bingham']   = bingham
    d['xy_mu']     = xy_mu
    d['xy_cov']    = xy_cov
    d['logzf_mu']  = logzf_mu
    d['logzf_cov'] = logzf_cov
    d['f']         = f
    return d


def set_xyz_labels(ax, z=True):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if z: ax.set_zlabel('z')


def plot_cam_pos(pos, ds_name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig.suptitle('Cam positions: ' + ds_name)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')
    set_xyz_labels(ax)


def plot_trans(trans, ds_name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    set_xyz_labels(ax)

    ax.title.set_text('Translations: ' + ds_name)
    ax.scatter(trans[:,0], trans[:,1], trans[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')


def plot_rot_axis(rot, axis, bingham, ds_name, plt_ax=None):
    if plt_ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        set_xyz_labels(ax)
        ax.title.set_text('Rotations ('+ ('x' if axis==0 else 'y' if axis==1 else 'z') + '-axis): ' + ds_name)
    else:
        ax = plt_ax

    x = np.array([0,0,0])
    x[axis] = 1

    pts = rot @ x

    if bingham is None:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='r', alpha=ALPHA, label='Fitted')
    else:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='b', alpha=ALPHA, label='Real')
        plot_rot_axis( 
            np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), bingham.random_samples(rot.shape[0])))),
            axis,
            None,
            ds_name + ' (Bingham)',
            ax if args.overlay else None)
    
    if plt_ax is None:
        ax.legend()


def plot_xy(xy, xy_mu, xy_cov, ds_name, plt_ax=None):
    if plt_ax is None:
        fig,ax = plt.subplots()
        set_xyz_labels(ax,z=False)
        ax.set_title('x:y: ' + ds_name)
        ax.add_patch(patches.Rectangle((-0.15, -0.15), 0.3, 0.3, alpha=ALPHA, color='k', label='Uniform'))
    else:
        ax = plt_ax

    if xy_mu is None:
        ax.scatter(xy[:,0], xy[:,1], color='r', alpha=ALPHA, label='Fitted')
    else:
        ax.scatter(xy[:,0], xy[:,1], color='b', alpha=ALPHA, label='Real')
        samples = nr.multivariate_normal(xy_mu, xy_cov, size=xy.shape[0])
        plot_xy(
            samples,
            None, None,
            ds_name,
            ax if args.overlay else None)

    if plt_ax is None:
        ax.legend()


def plot_zf(z, f, logzf_mu, logzf_cov, ds_name, plt_ax=None):
    if plt_ax is None:
        fig,ax = plt.subplots()
        ax.set_xlabel('z')
        ax.set_ylabel('f')
        ax.set_title('z-axis:focal_lenght: ' + ds_name)
        ax.add_patch(patches.Rectangle((0.8, 200), 1.6, 800, alpha=ALPHA, color='k', label='Uniform'))
    else:
        ax = plt_ax

    if logzf_mu is None:
        ax.scatter(z, f, color='r', alpha=ALPHA, label='Fitted')
    else:
        ax.scatter(z, f, color='b', alpha=ALPHA, label='Real')
        samples = np.exp(nr.multivariate_normal(logzf_mu, logzf_cov, size=z.shape[0]))
        plot_zf(
            samples[:,0],
            samples[:,1],
            None, None,
            ds_name,
            ax if args.overlay else None)

    if plt_ax is None:
        ax.legend()

def plot(args, d, ds_name):
    bingham, xy_mu, xy_cov, logzf_mu, logzf_cov = None, None, None, None, None
    if args.fit:
        bingham   = d['bingham']
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        logzf_mu  = d['logzf_mu']
        logzf_cov = d['logzf_cov']

    if args.zf:  plot_zf(d['t'][:,2], d['f'], logzf_mu, logzf_cov, ds_name)
    if args.xy:  plot_xy(d['t'][:,:2], xy_mu, xy_cov, ds_name)
    if args.cam: plot_cam_pos(d['cam_pos'], ds_name)
    if args.t:   plot_trans(d['t'], ds_name)
    if args.rx:  plot_rot_axis(d['R'], 0, bingham, ds_name)
    if args.ry:  plot_rot_axis(d['R'], 1, bingham, ds_name)
    if args.rz:  plot_rot_axis(d['R'], 2, bingham, ds_name)
    plt.show()

def get_outliers(data, q=0.05):
    med = np.median(data, axis=0)
    dist = np.sqrt(np.sum((data - med)**2, axis=-1))
    n = int(data.shape[0]*q)
    return np.argpartition(-dist, n)[:n]                                          


"""
def process_test(dataset, indices, outliers):
    d = process(dataset, outliers)

    for i in indices:
        TCO = np.linalg.inv(dataset[i][2]['camera']['TWC'])
        K = dataset[i][2]['camera']['K']
        R = TCO[:3,:3]
        t = TCO[:3,3]
        f = K[0,0]
        
        np.testing.assert_almost_equal(R, d['R'][i])
        np.testing.assert_almost_equal(t, d['t'][i])
        np.testing.assert_almost_equal(f, d['f'][i])

def test(args):
    pix3d = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'chair', not args.test)
    compcars = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', not args.test)
    standfordcars = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', not args.test)

    for dataset in [pix3d, compcars, standfordcars]:
        process_test(dataset, list(range(10)), args)
"""


if __name__ == '__main__':
    args = parser.parse_args([] if '__file__' not in globals() else None)
    if args.dataset is None:
        parser.print_help()
        exit(1)

    #test(args)

    if args.dataset[:5] == 'pix3d':
        c = args.dataset[6:]
        categories = pix3d_categories if c == '' else c.split(',') 

        for category in categories:
            dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category, not args.test)
            d = process(dataset, args.outliers)
            plot(args, d, 'pix3d-'+category)

    elif args.dataset == 'compcars':
        dataset = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', not args.test)
        d = process(dataset, args.outliers)
        plot(args, d, 'CompCars')

    elif args.dataset == 'stanfordcars':
        dataset = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', not args.test)
        d = process(dataset, args.outliers)
        plot(args, d, 'StanfordCars')

    else:
        parser.print_help()
        exit(1)
    