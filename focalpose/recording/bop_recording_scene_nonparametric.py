import numpy as np
import numpy.random as nr
import pinocchio as pin
from scipy.spatial.transform import Rotation

from focalpose.simulator import Camera
from focalpose.recording.bop_recording_scene import BopRecordingScene
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset

class BopRecordingSceneNonparametric(BopRecordingScene):
    def __init__(self,
                 deltas=None,
                 outliers = 0.05,

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
            self.real_dataset.index = self.real_dataset.index.drop(BopRecordingSceneNonparametric.get_outliers(zf, outliers))

        if deltas == None:
            q = 98
            xy = self.real_dataset.TWC[:,:2,3]
            zf = np.vstack([self.real_dataset.TWC[:,2,3], self.real_dataset.f]).T
            R = self.real_dataset.TWC[:,:3,:3]
            self.delta_x, self.delta_y = BopRecordingSceneNonparametric.get_delta(xy, q)
            self.delta_z, self.delta_f = BopRecordingSceneNonparametric.get_delta(zf, q)
            self.delta_R = BopRecordingSceneNonparametric.get_delta_rot(R, q)
        else:
            self.delta_x = deltas['x']
            self.delta_y = deltas['y']
            self.delta_z = deltas['z']
            self.delta_f = deltas['f']
            self.delta_R = deltas['R']

    def sample_camera(self):
        i = nr.randint(0, len(self.real_dataset))

        dx,dy = BopRecordingSceneNonparametric.sample_from_unit_sphere(2) * np.array([self.delta_x,self.delta_y])
        dz,df = BopRecordingSceneNonparametric.sample_from_unit_sphere(2) * np.array([self.delta_z,self.delta_f])
        dR = Rotation.from_rotvec(BopRecordingSceneNonparametric.sample_from_unit_sphere(3) * self.delta_R).as_matrix()

        f = self.real_dataset.f[i] + df
        TWC = self.real_dataset.TWC[i]
        x = TWC[0,3] + dx
        y = TWC[1,3] + dy
        z = TWC[2,3] + dz
        R = TWC[:3,:3] @ dR
        Rt = np.hstack([R,np.array([[x],[y],[z]])])
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

    def nearest_dists(data):
        dists = []
        for i in range(data.shape[0]):
            if len(data.shape) == 1:
                dist2 = (data[i]-data)**2
            else:
                dist2 = np.sum((data[i]-data)**2, axis=-1)
            dists.append(np.sqrt(dist2[np.argpartition(dist2, 1)[1]]))
        return dists

    def nearest_dists_rot(data):
        dists = []
        for i in range(data.shape[0]):
            R_deltas = data[i].T @ data
            dist2 = np.arccos( np.clip((np.trace(R_deltas, axis1=1, axis2=2)-1)/2, -1, 1) )
            dists.append( dist2[np.argpartition(dist2, 1)[1]] )
        return dists

    def get_delta(data, q):
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        ranges = maxs - mins
        ranges[ranges==0] = 1
        data_norm = (data - mins) / ranges
        delta = np.percentile(BopRecordingSceneNonparametric.nearest_dists(data_norm), q) 
        return delta * ranges

    def get_delta_rot(data, q):
        return np.percentile(BopRecordingSceneNonparametric.nearest_dists_rot(data), q)

    def sample_from_unit_sphere(dim):
        rng = np.random.default_rng()
        X = rng.normal(size=(dim))
        U = rng.random(1) 
        return U**(1/dim) / np.sqrt(np.sum(X**2, keepdims=True)) * X

    def get_outliers(data, q=0.05):
        med = np.median(data, axis=0)
        dist = np.sqrt(np.sum((data - med)**2, axis=-1))
        n = int(data.shape[0]*q)
        return np.argpartition(-dist, n)[:n]                                          
