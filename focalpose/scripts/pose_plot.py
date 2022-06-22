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
#parser.add_argument('--dataset',  type=str,   default=None, help='{stanfordcars, compcars, pix3d-{bed, chair, sofa, table} }')
parser.add_argument('--outliers', type=float, default=0.05,  help='Portion of points to remove from dataset as outliers.')
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

parser.add_argument('--alpha', type=float, default=0.25, help="The alpha blending value of plot objects.")


pix3d_categories = ['bed', 'chair', 'sofa', 'table']
FIGSIZE_2D=(8,6)
FIGSIZE_3D=(7,6)
LEGEND_LOC='upper right'


def process(dataset, outliers):
    if outliers > 0:
        d = process(dataset, 0)
        t = dataset.TCO[:,:3,3]
        zf = np.vstack([t[:,2], dataset.f]).T
        dataset.index = dataset.index.drop(get_outliers(zf, outliers))

    d = dict()
    mat = dataset.TCO
    R = mat[:,:3,:3]
    t = mat[:,:3, 3]
    f = dataset.f
    cam_poses = (-R@t[:, :, None]).squeeze()

    R_quat = np.array(list(  map(lambda x: Rotation.from_matrix(x).as_quat(), R)  ))
    bingham = BinghamDistribution.fit(R_quat)

    xy = t[:,:2]
    xy_mu = np.mean(xy, axis=0)
    xy_cov = np.cov(xy.T)

    zf = np.vstack([t[:,2],f]).T
    logzf = np.log(zf)
    zf_log_mu = np.mean(logzf, axis=0)
    zf_log_cov = np.cov(logzf.T)

    d['R']         = R
    d['t']         = t
    d['cam_pos']   = cam_poses
    d['R_quat']    = R_quat

    #d['bingham']   = bingham
    d['bingham_z'] = bingham._param_z
    d['bingham_m'] = bingham._param_m

    d['xy_mu']     = xy_mu
    d['xy_cov']    = xy_cov

    d['zf_log_mu']  = zf_log_mu
    d['zf_log_cov'] = zf_log_cov

    d['f']         = f
    return d


def set_xyz_labels(ax, z=True):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if z: ax.set_zlabel('z')


def plot_cam_pos(pos, ds_name):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = plt.axes(projection='3d')
    fig.suptitle('Cam positions: ' + ds_name)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')
    set_xyz_labels(ax)


def plot_trans(trans, ds_name):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = plt.axes(projection='3d')
    set_xyz_labels(ax)

    ax.title.set_text('Translations: ' + ds_name)
    ax.scatter(trans[:,0], trans[:,1], trans[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')


def plot_rot_axis(rot, axis, bingham, ds_name, overlay):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = plt.axes(projection='3d')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    set_xyz_labels(ax)
    ax.title.set_text('Rotations ('+ ('x' if axis==0 else 'y' if axis==1 else 'z') + '-axis): ' + ds_name)
    
    unit_vector = np.array([0,0,0])
    unit_vector[axis] = 1
    pts = rot @ unit_vector

    if bingham is not None:
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='b', alpha=args.alpha, label='Real')
        
        sample_rot = np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), bingham.random_samples(rot.shape[0]))))
        sample_pts = sample_rot @ unit_vector
        if overlay:
            ax.scatter(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2], color='r', alpha=args.alpha, label='Fitted')
        else:
            plot_rot_axis(sample_rot, axis, None, ds_name, True)
    else:
        if overlay: ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='r', alpha=args.alpha, label='Fitted')
        else:       ax.scatter(pts[:,0], pts[:,1], pts[:,2], color='b', alpha=args.alpha, label='Real')

    ax.legend(loc=LEGEND_LOC)


def plot_xy(xy, xy_mu, xy_cov, ds_name, overlay):
    fig,ax = plt.subplots(figsize=FIGSIZE_2D)
    set_xyz_labels(ax,z=False)
    ax.set_title('x:y : ' + ds_name)
    ax.add_patch(patches.Rectangle((-0.15, -0.15), 0.3, 0.3, alpha=args.alpha, color='k', label='FocalPose synt. data'))
    
    if xy_mu is not None:
        ax.scatter(xy[:,0], xy[:,1], color='b', alpha=args.alpha, label='Real')
        
        samples = nr.multivariate_normal(xy_mu, xy_cov, size=xy.shape[0])
        if overlay:
            ax.scatter(samples[:,0], samples[:,1], color='r', alpha=args.alpha, label='Fitted')
        else:
            plot_xy(samples, None, None, ds_name, True)
    else:
        if overlay: ax.scatter(xy[:,0], xy[:,1], color='r', alpha=args.alpha, label='Fitted')
        else:       ax.scatter(xy[:,0], xy[:,1], color='b', alpha=args.alpha, label='Real')

    ax.legend(loc=LEGEND_LOC)


def plot_zf(z, f, zf_log_mu, zf_log_cov, ds_name, overlay):
    fig,ax = plt.subplots(figsize=FIGSIZE_2D)
    ax.set_xlabel('z')
    ax.set_ylabel('f')
    ax.set_title('z:f : ' + ds_name)
    
    if ds_name == 'pix3d-chair':  z_interval = (0.8, 3.4)  
    if ds_name == 'stanfordcars': z_interval = (0.8, 3.0)
    if ds_name == 'compcars':     z_interval = (0.8, 3.0)
    else:                         z_interval = (0.8, 2.4)
    
    ax.add_patch(patches.Rectangle((z_interval[0], 200), z_interval[1]-z_interval[0], 800, alpha=args.alpha, color='k', label='FocalPose synt. data'))

    if zf_log_mu is not None:
        ax.scatter(z, f, color='b', alpha=args.alpha, label='Real')
        
        samples = np.exp(nr.multivariate_normal(zf_log_mu, zf_log_cov, size=z.shape[0]))
        if overlay:
            ax.scatter(samples[:,0], samples[:,1], color='r', alpha=args.alpha, label='Fitted')
        else:
            plot_zf(samples[:,0], samples[:,1], None, None, ds_name, True)
    else:
        if overlay: ax.scatter(z, f, color='r', alpha=args.alpha, label='Fitted')
        else:       ax.scatter(z, f, color='b', alpha=args.alpha, label='Real')

    ax.legend(loc=LEGEND_LOC)


def plot(args, d, ds_name):
    bingham, xy_mu, xy_cov, zf_log_mu, zf_log_cov = None, None, None, None, None
    if args.fit:
        bingham   = BinghamDistribution(d['bingham_m'], d['bingham_z'])
        xy_mu     = d['xy_mu']
        xy_cov    = d['xy_cov']
        zf_log_mu  = d['zf_log_mu']
        zf_log_cov = d['zf_log_cov']

    if args.zf:  plot_zf(d['t'][:,2], d['f'], zf_log_mu, zf_log_cov, ds_name, args.overlay)
    if args.xy:  plot_xy(d['t'][:,:2], xy_mu, xy_cov, ds_name, args.overlay)
    if args.cam: plot_cam_pos(d['cam_pos'], ds_name)
    if args.t:   plot_trans(d['t'], ds_name)
    if args.rx:  plot_rot_axis(d['R'], 0, bingham, ds_name, args.overlay)
    if args.ry:  plot_rot_axis(d['R'], 1, bingham, ds_name, args.overlay)
    if args.rz:  plot_rot_axis(d['R'], 2, bingham, ds_name, args.overlay)
    plt.show()


def get_outliers(data, q=0.05):
    med = np.median(data, axis=0)
    dist = np.sqrt(np.sum((data - med)**2, axis=-1))
    n = int(data.shape[0]*q)
    return np.argpartition(-dist, n)[:n]                                          


if __name__ == '__main__':
    args = parser.parse_args([] if '__file__' not in globals() else None)
    if not args.fit and args.overlay:
        print("Error: cannot overlay plots if --fit=False")
        parser.print_help()
        exit(1)

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
    