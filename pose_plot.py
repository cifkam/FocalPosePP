#!/usr/bin/env python

from sys import exit
import argparse
import numpy as np
from pathlib import Path
from matplotlib import projections, pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pyquaternion import Quaternion

from deep_bingham.bingham_distribution import BinghamDistribution
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset

import numpy.linalg

parser = argparse.ArgumentParser()
parser.add_argument('--cam', action="store_true", help="Plot camera positions")
parser.add_argument('--t', action="store_true", help="Plot translations")
parser.add_argument('--rx', action="store_true", help="Plot rotation of x-axes")
parser.add_argument('--ry', action="store_true", help="Plot rotation of y-axes")
parser.add_argument('--rz', action="store_true", help="Plot rotation of z-axes")
parser.add_argument('--bingham', action="store_true", help="Plot samples from bingham distribution of rotation")
parser.add_argument('--dataset', default=None, type=str, help="{stanfordcars | compcars, pix3d-{bed, chair, sofa, table} }")

parser.add_argument('--xy', action="store_true")
parser.add_argument('--zf', action="store_true")
parser.add_argument('--train', action="store_true")
parser.set_defaults(cam=False, t=False, rx=False, ry=False, rz=False, bingham=False, xy=False, zf=False, train=False)

focalpose_categories = ['bed', 'chair', 'sofa', 'table']


def process(dataset):
    d = dict()
    R = dataset.R
    t = dataset.t
    f = dataset.f
    cam_poses = (-R@t[:, :, None]).squeeze()

    R_quat = np.array(list(  map(lambda x: Quaternion._from_matrix(x), R)  ))
    R_quat_np = np.array(list(  map(lambda x: x.q, R_quat)  ))
    bingham = BinghamDistribution.fit(R_quat_np)

    xy = t[:,:2]
    xy_mean = np.mean(xy, axis=0)
    xy_cov = (xy-xy_mean).T @ (xy-xy_mean)

    d['R']         = R
    d['t']         = t
    d['cam_pos']   = cam_poses
    d['R_quat']    = R_quat
    d['R_quat_np'] = R_quat_np
    d['bingham']   = bingham
    d['xy_mean']   = xy_mean
    d['xy_cov']    = xy_cov
    d['f']         = f
    return d


def process_test(dataset, indices):
    d = process(dataset)

    for i in indices:
        TCO = np.linalg.inv(dataset[i][2]['camera']['TWC'])
        K = dataset[i][2]['camera']['K']
        R = TCO[:3,:3]
        t = TCO[:3,3]
        f = K[0,0]
        
        np.testing.assert_almost_equal(R, d['R'][i])
        np.testing.assert_almost_equal(t, d['t'][i])
        np.testing.assert_almost_equal(f, d['f'][i])


def set_ax_labels(ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_cam_pos(pos, ds_name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig.suptitle("Cam positions: " + ds_name)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')
    set_ax_labels(ax)


def plot_trans(trans, ds_name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    set_ax_labels(ax)

    ax.title.set_text("Translations: " + ds_name)
    ax.scatter(trans[:,0], trans[:,1], trans[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')


def plot_rot_axis(rot, axis, ds_name, bingham=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    set_ax_labels(ax)

    x = np.array([0,0,0])
    x[axis] = 1

    ax.title.set_text("Rotations ("+ ("x" if axis==0 else "y" if axis==1 else "z") + "-axis): " + ds_name)
    pts = rot @ x
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='b')
    ax.scatter([0], [0], [0], c='r')

    if bingham is not None:
        plot_rot_axis( 
            np.array(list(map(lambda x: Quaternion(x).rotation_matrix, bingham.random_samples(rot.shape[0])))),
            axis,
            ds_name + ' (Bingham)')

def plot_xy(d, ds_name):
    fig,ax = plt.subplots()
    ax.scatter(d['t'][:,0], d['t'][:,1])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('x:y: ' + ds_name)


def plot_zf(d, ds_name):
    fig,ax = plt.subplots()
    ax.scatter(d['t'][:,2], d['f'])

    ax.set_xlabel('z')
    ax.set_ylabel('f')
    ax.set_title('z-axis:focal_lenght: ' + ds_name)


def plot(args, d, ds_name):
    bingham = d['bingham'] if args.bingham else None

    if args.zf:  plot_zf(d, ds_name)
    if args.xy:  plot_xy(d, ds_name)
    if args.cam: plot_cam_pos(d['cam_pos'], ds_name)
    if args.t:   plot_trans(d['t'], ds_name)
    if args.rx:  plot_rot_axis(d['R'], 0, ds_name, bingham)
    if args.ry:  plot_rot_axis(d['R'], 1, ds_name, bingham)
    if args.rz:  plot_rot_axis(d['R'], 2, ds_name, bingham)
    plt.show()


def test(args):
    pix3d = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'chair', args.train)
    compcars = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', args.train)
    standfordcars = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', args.train)

    for dataset in [pix3d, compcars, standfordcars]:
        process_test(dataset, list(range(10)))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    if args.dataset is None:
        parser.print_help()
        exit(1)

    #test(args)

    if args.dataset[:5] == 'pix3d':
        c = args.dataset[6:]
        categories = focalpose_categories if c == '' else c.split(',') 

        for category in categories:
            dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', category, args.train)
            d = process(dataset)
            plot(args, d, 'pix3d-'+category)

    elif args.dataset == 'compcars':
        dataset = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', args.train)
        d = process(dataset)
        plot(args, d, 'CompCars')

    elif args.dataset == 'stanfordcars':
        dataset = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', args.train)
        d = process(dataset)
        plot(args, d, 'StanfordCars')

    else:
        parser.print_help()
        exit(1)
    