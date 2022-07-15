import numpy as np
import numpy.random as nr
import pinocchio as pin

from deep_bingham.bingham_distribution import BinghamDistribution
from scipy.spatial.transform import Rotation

from focalpose.simulator import Camera
from focalpose.recording.bop_recording_scene import BopRecordingScene

class BopRecordingSceneParametric(BopRecordingScene):
    def __init__(self,

                 xy_mu,
                 xy_cov,
                 zf_log_mu,
                 zf_log_cov,
                 rot_bingham_z,
                 rot_bingham_m,
                 
                 urdf_ds='ycbv',
                 texture_ds='shapenet',
                 domain_randomization=True,
                 background_cage=True,
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
            background_cage=background_cage,
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

        # Camera params
        self.xy_mu = xy_mu
        self.xy_cov = xy_cov
        self.zf_log_mu = zf_log_mu
        self.zf_log_cov = zf_log_cov
        self.rot_bingham_z = rot_bingham_z
        self.rot_bingham_m = rot_bingham_m

    def sample_camera(self):
        x,y = nr.multivariate_normal(self.xy_mu, self.xy_cov)
        z,f = np.exp( nr.multivariate_normal(self.zf_log_mu, self.zf_log_cov) )

        q = BinghamDistribution(np.array(self.rot_bingham_m), np.array(self.rot_bingham_z)).random_samples(1)
        R = Rotation.from_quat(q).as_matrix().reshape((3,3))
        t = np.array([[x,y,z]]).T
        TWC = np.vstack( [np.hstack([R.T, (-R.T@t).reshape(-1,1)]), [0,0,0,1]] )

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