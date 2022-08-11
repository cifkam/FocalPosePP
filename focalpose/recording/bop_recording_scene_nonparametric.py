import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from focalpose.simulator import Camera
from focalpose.recording.bop_recording_scene import BopRecordingScene
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset
from focalpose.fitting.nonparametric_model import NonparametricModel
from focalpose.fitting.fitting import get_outliers

class BopRecordingSceneNonparametric(BopRecordingScene):
    def __init__(self,
                 deltas=None,
                 outliers = 0.05,
                 q=95,

                 urdf_ds='ycbv',
                 texture_ds='shapenet',
                 domain_randomization=True,
                 background_textures=True,
                 textures_on_objects=False,
                 n_objects_interval=(1, 1),
                 #objects_xyz_interval=((0.0, -0.5, -0.15), (1.0, 0.5, 0.15)),
                 proba_falling=0.0,
                 resolution=(640, 480),
                 #focal_interval=(515, 515),
                 #camera_distance_interval=(0.5, 1.5),
                 border_check=True,
                 gpu_renderer=True,
                 n_textures_cache=50,
                 seed=0):

        super().__init__(
            urdf_ds=urdf_ds,
            texture_ds=texture_ds,
            domain_randomization=domain_randomization,
            background_textures=background_textures,
            textures_on_objects=textures_on_objects,
            n_objects_interval=n_objects_interval,
            #objects_xyz_interval=objects_xyz_interval,
            proba_falling=proba_falling,
            resolution=resolution,
            #focal_interval=focal_interval,
            #camera_distance_interval=camera_distance_interval,
            border_check=border_check,
            gpu_renderer=gpu_renderer,
            n_textures_cache=n_textures_cache,
            seed=seed)

        if urdf_ds == 'pix3d-sofa':
            self.real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'sofa')
        elif urdf_ds == 'pix3d-bed':
            self.real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'bed')
        elif urdf_ds == 'pix3d-table':
            self.real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'table')
        elif 'pix3d-chair' in urdf_ds:
            self.real_dataset = Pix3DDataset(LOCAL_DATA_DIR / 'pix3d', 'chair')
        elif 'stanfordcars' in urdf_ds:
            self.real_dataset = StanfordCars3DDataset(LOCAL_DATA_DIR / 'StanfordCars')
        elif 'compcars' in urdf_ds:
            self.real_dataset = CompCars3DDataset(LOCAL_DATA_DIR / 'CompCars')

        if outliers > 0:
            t = self.real_dataset.TCO[:,:3,3]
            zf = np.vstack([t[:,2], self.real_dataset.f]).T
            self.real_dataset.index = self.real_dataset.index.drop(get_outliers(zf, outliers))

        if deltas is None:
            self.nonparametric_model = NonparametricModel.fit(self.real_dataset, q)
        else:
            self.nonparametric_model = NonparametricModel(
                self.real_dataset,
                deltas['R'],
                deltas['x'],
                deltas['y'],
                deltas['z'],
                deltas['f'])

    def sample_camera(self):
        R,t,f = self.nonparametric_model.sample()
        Rt = np.hstack([R,t.reshape(-1,1)])
        TWC = np.vstack([ Rt , [0,0,0,1] ])

        K = np.zeros((3, 3), dtype=np.float)
        W, H = max(self.resolution), min(self.resolution)
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = W / 2
        K[1, 2] = H / 2
        K[2, 2] = 1.0
        cam = Camera(resolution=self.resolution, client_id=self._client_id)
        cam.set_intrinsic_K(K)
        cam.set_extrinsic_T(TWC)
        return cam
        
    def objects_pos_orn_rand(self):
        self.hide_plane()
        for body in self.bodies:
            pos = np.zeros(3)
            orn = pin.Quaternion().coeffs()
            body.pose = pos, orn