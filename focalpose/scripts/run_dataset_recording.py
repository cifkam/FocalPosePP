import argparse
from colorama import Fore, Style
import json

from focalpose.config import LOCAL_DATA_DIR
from focalpose.recording.record_dataset import record_dataset
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset

def make_cfg(cfg_name,
             resume_ds_name='',
             debug=False,
             distributed=False,
             overwrite=False,
             datasets_dir=LOCAL_DATA_DIR,
             pose=''):

    datasets_dir = datasets_dir / 'synt_datasets'
    datasets_dir.mkdir(exist_ok=True)

    cfg = argparse.ArgumentParser('').parse_args([])

    cfg.overwrite = overwrite
    cfg.ds_name = 'default_dataset'

    n_frames = 1e6
    cfg.n_frames_per_chunk = 100
    cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
    cfg.train_ratio = 0.95

    cfg.distributed = distributed
    cfg.n_workers = 12
    cfg.n_processes_per_gpu = 10

    cfg.scene_cls = 'focalpose.recording.bop_recording_scene.BopRecordingScene'
    cfg.scene_kwargs = dict(
        gpu_renderer=True,
        texture_ds='shapenet',
        domain_randomization=True,
        n_objects_interval=(1, 1),
        proba_falling=0.0,
        border_check=True,
        n_textures_cache=100,
        objects_xyz_interval=((-0.15, -0.15, 0.), (0.15, 0.15, 0.)),
        camera_distance_interval=(0.8, 2.4),
        focal_interval=(200, 1000),
        resolution=(640, 640),
        textures_on_objects=True,
    )
    cfg.ds_name = f'{cfg_name}-1M'

    if cfg_name == 'pix3d-sofa':
        n_frames = 1e6
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds='pix3d-sofa',
        )
        fit_file = LOCAL_DATA_DIR / 'pix3d' / 'sofa-fit.json'
        real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'sofa', True)

    elif cfg_name == 'pix3d-bed':
        n_frames = 1e6
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds='pix3d-bed',
        )
        fit_file = LOCAL_DATA_DIR / 'pix3d' / 'bed-fit.json'
        real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'bed', True)

    elif cfg_name == 'pix3d-table':
        n_frames = 1e6
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds='pix3d-table',
        )
        fit_file = LOCAL_DATA_DIR / 'pix3d' / 'table-fit.json'
        real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'table', True)

    elif 'pix3d-chair' in cfg_name:
        cfg.scene_kwargs['camera_distance_interval'] = (0.8, 3.4)
        n_frames = 50000
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds=cfg_name,
        )
        fit_file = LOCAL_DATA_DIR / 'pix3d' / 'chair-fit.json'
        real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'chair', True)

    elif cfg_name == 'pix3d':
        n_frames = 1e6
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds='pix3d',
        )
        if pose != 'uniform':
            raise NotImplementedError('')

    elif 'stanfordcars' in cfg_name:
        n_frames = 75000
        cfg.scene_kwargs['camera_distance_interval'] = (0.8, 3.0)
        cfg.scene_kwargs['resolution'] = (300, 200)
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds=cfg_name,
        )
        fit_file = LOCAL_DATA_DIR / 'StanfordCars' / 'fit.json'
        real_dataset = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars', True)
        
    elif 'compcars' in cfg_name:
        n_frames = 100000
        cfg.scene_kwargs['camera_distance_interval'] = (0.8, 3.0)
        cfg.scene_kwargs['resolution'] = (300, 200)
        cfg.n_frames_per_chunk = 100
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk
        cfg.ds_name = f'{cfg_name}-1M'

        cfg.scene_kwargs.update(
            urdf_ds=cfg_name,
        )
        fit_file = LOCAL_DATA_DIR / 'CompCars' / 'fit.json'
        real_dataset = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars', True)
        
    elif resume_ds_name:
        pass

    else:
        raise ValueError('Unknown config')

    
    if pose == 'uniform':
        pass

    elif pose == 'real':
        cfg.scene_cls = 'focalpose.recording.bop_recording_scene_realpose.BopRecordingSceneRealPose'
        cfg.scene_kwargs.pop('objects_xyz_interval')
        cfg.scene_kwargs.pop('camera_distance_interval')
        cfg.scene_kwargs.pop('focal_interval')
        cfg.scene_kwargs['real_dataset'] = real_dataset
        
    elif pose == 'parametric':
        with open(fit_file) as f:
                fit_json = json.loads(f.read())
        cfg.scene_cls = 'focalpose.recording.bop_recording_scene_parametric.BopRecordingSceneParametric'
        cfg.scene_kwargs.pop('objects_xyz_interval')
        cfg.scene_kwargs.pop('camera_distance_interval')
        cfg.scene_kwargs.pop('focal_interval')
        cfg.scene_kwargs['xy_mu'] = fit_json['xy_mu']
        cfg.scene_kwargs['xy_cov'] = fit_json['xy_cov']
        cfg.scene_kwargs['zf_log_mu'] = fit_json['zf_log_mu']
        cfg.scene_kwargs['zf_log_cov'] = fit_json['zf_log_cov']
        cfg.scene_kwargs['rot_bingham_z'] = fit_json['rot_bingham_z']
        cfg.scene_kwargs['rot_bingham_m'] = fit_json['rot_bingham_m']
    else:
        raise ValueError('Unknown pose mode')
        

    if debug:
        n_frames = 10
        cfg.overwrite = True
        cfg.ds_name = 'debug'
        cfg.n_frames_per_chunk = 1
        cfg.n_chunks = n_frames // cfg.n_frames_per_chunk

    if resume_ds_name:
        cfg.resume = datasets_dir / resume_ds_name
        cfg.ds_name = resume_ds_name
        assert cfg.resume.exists()
    else:
        cfg.resume = ''
        cfg.ds_dir = datasets_dir / cfg.ds_name
    return cfg


def main():
    parser = argparse.ArgumentParser('Dataset recording')
    parser.add_argument('--config', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--pose', default='uniform', type=str, help="{ uniform | parametric | real }")

    args = parser.parse_args()

    print(f"{Fore.RED}using config {args.config} {Style.RESET_ALL}")
    cfg = make_cfg(args.config,
                   resume_ds_name=args.resume,
                   debug=args.debug,
                   distributed=not args.local,
                   overwrite=args.overwrite,
                   pose=args.pose)
    for k, v in vars(cfg).items():
        print(k, v)

    if cfg.resume:
        print(f"RESUMING {Fore.RED} {cfg.ds_name} {Style.RESET_ALL} \n ")
    else:
        print(f"STARTING DATASET RECORDING {Fore.GREEN} {cfg.ds_name} {Style.RESET_ALL} \n ")

    record_dataset(cfg)


if __name__ == '__main__':
    main()
