import os
import shutil
import argparse
import numpy as np
from colorama import Fore, Style

from focalpose.utils.resources import assign_gpu
from focalpose.training.train_pose import train_pose
from focalpose.utils.logging import get_logger
logger = get_logger(__name__)

def make_cfg(args):
    cfg = argparse.ArgumentParser('').parse_args([])
    if args.config:
        logger.info(f"{Fore.GREEN}Training with config: {args.config} {Style.RESET_ALL}")

    cfg.resume_run_id = None
    if len(args.resume) > 0:
        cfg.resume_run_id = args.resume
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")

    if 'SLURM_PROCID' in os.environ:
        N_CPUS = int(os.environ.get('SLURM_PROCID', 10))
    elif 'MPI_LOCALRANKID' in os.environ:
        N_CPUS = int(os.environ.get('MPI_LOCALRANKID', 10))
    else:
        N_CPUS = 8
    N_WORKERS = min(N_CPUS - 2, 8)
    N_WORKERS = 8
    N_RAND = np.random.randint(1e6)

    run_comment = ''

    # Data
    if 'pix3d-sofa-bed-table' in args.config:
        cfg.urdf_ds_name = 'pix3d-sofa-bed-table'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-bed-table-1M.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None),
                                ('pix3d-sofa.test', 1, None),
                                ('pix3d-bed.test', 1, None)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-bed-table-1M.train', 1, None),
                                  ('pix3d-table.train', 1, None),
                                  ('pix3d-sofa.train', 1, None),
                                  ('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None),
                                ('pix3d-sofa.test', 1, None),
                                ('pix3d-bed.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-table.train', 1, None),
                                  ('pix3d-sofa.train', 1, None),
                                  ('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None),
                                ('pix3d-sofa.test', 1, None),
                                ('pix3d-bed.test', 1, None)]

    elif 'pix3d-sofa' in args.config:
        cfg.urdf_ds_name = 'pix3d-sofa'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 0, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F005p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 1, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 10, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F1p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 20, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F2p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 40, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F5p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 100, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-sofa-1M.train', 1, None), ('pix3d-sofa.train', 200, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-sofa.train', 1, None)]
            cfg.test_ds_names = [('pix3d-sofa.test', 1, None)]

    elif 'pix3d-bed' in args.config:
        cfg.urdf_ds_name = 'pix3d-bed'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'F005p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, None), ('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, None), ('pix3d-bed.train', 25, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'F1p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, None), ('pix3d-bed.train', 50, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'F2p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, None), ('pix3d-bed.train', 100, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-bed-1M.train', 1, None), ('pix3d-bed.train', 500, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-bed.train', 1, None)]
            cfg.test_ds_names = [('pix3d-bed.test', 1, None)]

    elif 'pix3d-table' in args.config:
        cfg.urdf_ds_name = 'pix3d-table'
        cfg.n_symmetries_batch = 1

        if 'synth' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'F005p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, None), ('pix3d-table.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'F05p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, None), ('pix3d-table.train', 20, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'F1p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, None), ('pix3d-table.train', 40, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'F2p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, None), ('pix3d-table.train', 80, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'F10p' in args.config:
            cfg.train_ds_names = [('synthetic.pix3d-table-1M.train', 1, None), ('pix3d-table.train', 400, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]
        elif 'real' in args.config:
            cfg.train_ds_names = [('pix3d-table.train', 1, None)]
            cfg.test_ds_names = [('pix3d-table.test', 1, None)]

    elif 'pix3d-chair-1' in args.config:
        cfg.urdf_ds_name = 'pix3d-chair-p1'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [(f'synthetic.pix3d-chair-{i}-1M.train', 1, None) for i in range(1,5)] +\
                                 [(f'pix3d-chair-{i}.train', 3, None) for i in range(1,5)]
            cfg.test_ds_names =  [(f'pix3d-chair-{i}.test', 1, None) for i in range(1,5)]

    elif 'pix3d-chair-2' in args.config:
        cfg.urdf_ds_name = 'pix3d-chair-p2'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [(f'synthetic.pix3d-chair-{i}-1M.train', 1, None) for i in range(5,10)] +\
                                 [(f'pix3d-chair-{i}.train', 3, None) for i in range(5,10)]
            cfg.test_ds_names =  [(f'pix3d-chair-{i}.test', 1, None) for i in range(5,10)]

    elif 'pix3d-chair' in args.config:
        cfg.urdf_ds_name = 'pix3d-chair'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [(f'synthetic.pix3d-chair-{i}-1M.train', 1, None) for i in range(1,22)] +\
                                 [('pix3d-chair.train', 4, None)]
            cfg.test_ds_names =  [('pix3d-chair.test', 1, None)]

    elif 'compcars3d' in args.config:
        cfg.urdf_ds_name = 'compcars3d'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [(f'synthetic.compcars3d-{i}-1M.train', 1, None) for i in range(1,11)] +\
                                 [('compcars3d.train', 3, None)]
            cfg.test_ds_names =  [('compcars3d.test', 1, None)]

    elif 'stanfordcars3d' in args.config:
        cfg.urdf_ds_name = 'stanfordcars3d'
        cfg.n_symmetries_batch = 1

        if 'F05p' in args.config:
            cfg.train_ds_names = [(f'synthetic.stanfordcars3d-{i}-1M.train', 1, None) for i in range(1,14)] +\
                                 [('stanfordcars3d.train', 1, None)]
            cfg.test_ds_names =  [('stanfordcars3d.test', 1, None)]

    cfg.eval_epoch_interval = 10
    if 'stanfordcars3d' in args.config:
        cfg.input_resize = (200, 300)
    elif 'compcars3d' in args.config:
        cfg.input_resize = (200, 300)
    else:
        cfg.input_resize = (640, 640)
    cfg.rgb_augmentation = True
    cfg.background_augmentation = True
    cfg.gray_augmentation = False

    # Model
    cfg.backbone_str = 'resnet50'
    cfg.backbone_pretrained = True
    cfg.run_id_pretrain = None
    cfg.n_pose_dims = 9
    cfg.n_rendering_workers = 16
    cfg.refiner_run_id_for_test = None
    cfg.coarse_run_id_for_test = None

    # Optimizer
    cfg.lr = 3e-4
    cfg.n_epochs_warmup = 50
    cfg.lr_epoch_decay = 500
    if 'wd' in args.config:
        cfg.weight_decay = 1e-4
    else:
        cfg.weight_decay = 0
    cfg.clip_grad_norm = 0.5

    # Training
    cfg.batch_size = 32
    cfg.epoch_size = 115200
    cfg.n_epochs = 700
    cfg.n_dataloader_workers = N_WORKERS

    # Method
    cfg.loss_disentangled = True
    cfg.n_points_loss = 2600
    cfg.n_iterations = 1
    cfg.min_area = None
    cfg.predict_focal_length = True
    cfg.new_update_rule = args.new_update_rule
    if 'disent' in args.config:
        cfg.loss_f_type = 'huber+reprojection+disent'
    elif 'reproj' in args.config:
        cfg.loss_f_type = 'huber+reprojection'
    elif 'huber' in args.config:
        cfg.loss_f_type = 'huber'
    cfg.loss_f_lambda = 1e-2
    cfg.perturb_bboxes = True

    if '-coarse' in args.config:
        cfg.TCO_input_generator = 'fixed'
    elif '-refine' in args.config:
        cfg.TCO_input_generator = 'gt+noise'
    else:
        raise ValueError("Unknown config: ", args.config)
    cfg.run_id = f'{args.config}-{run_comment}-{N_RAND}'

    if args.debug:
        cfg.n_epochs = 4
        cfg.eval_epoch_interval = 1
        cfg.batch_size = 32
        cfg.epoch_size = 16 * cfg.batch_size
        cfg.run_id = 'debug-' + cfg.run_id
        cfg.background_augmentation = True
        cfg.n_dataloader_workers = 8
        cfg.n_rendering_workers = 0
        cfg.n_test_frames = 10

    if 'SLURM_NTASKS' in os.environ:
        N_GPUS = int(os.environ.get('SLURM_NTASKS', 1))
    elif 'PMI_SIZE' in os.environ:
        N_GPUS = int(os.environ.get('PMI_SIZE', 1))
    else:
        N_GPUS = 1
    cfg.epoch_size = cfg.epoch_size // N_GPUS
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--new-update-rule', action='store_true')
    args = parser.parse_args()
    assign_gpu()
    cfg = make_cfg(args)
    train_pose(cfg)
