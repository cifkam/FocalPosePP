import numpy as np
import numpy.random as nr
import pinocchio as pin

from deep_bingham.bingham_distribution import BinghamDistribution
from scipy.spatial.transform import Rotation

from focalpose.simulator import Camera
from focalpose.recording.bop_recording_scene import BopRecordingScene, SamplerError
from focalpose.lib3d import Transform

class BopRecordingSceneNonuniform(BopRecordingScene):
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
            textures_on_objects=textures_on_objects,
            n_objects_interval=n_objects_interval,
            #None, # objects_xyz_interval,
            proba_falling=proba_falling,
            resolution=resolution,
            #None, # focal_interval,
            #None, # camera_distance_interval,
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
        #TCO = np.vstack([ np.hstack([R,t]), np.array([[0,0,0,1]]) ])
        #TWC = np.linalg.inv(TCO)
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
        
    def camera_rand(self):
        N = 0
        valid = False
        self.cam_obs = None
        while not valid:
            cam = self.sample_camera()
            cam_obs_ = cam.get_state()

            mask = cam_obs_['mask']
            mask[mask == self.background._body_id] = 0
            mask[mask == 255] = 0
            uniqs = np.unique(cam_obs_['mask'])

            valid = len(uniqs) == len(self.bodies) + 1
            if valid and self.border_check:
                for uniq in uniqs[uniqs > 0]:
                    H, W = cam_obs_['mask'].shape
                    ids = np.where(cam_obs_['mask'] == uniq)
                    if ids[0].max() == H-1 or ids[0].min() == 0 or \
                       ids[1].max() == W-1 or ids[1].min() == 0:
                        valid = False
            N += 1
            if N >= 3:
                raise SamplerError('Cannot sample valid camera configuration.')
            self.cam_obs = cam_obs_

    def objects_pos_orn_rand(self):
        self.hide_plane()
        for body in self.bodies:
            pos = np.zeros(3)
            orn = pin.Quaternion().coeffs()
            body.pose = pos, orn
    
    def _full_rand(self,
                   objects=True,
                   objects_pos_orn=True,
                   falling=False,
                   background_pos_orn=True,
                   camera=True,
                   visuals=True):
        if background_pos_orn:
            self.background_pos_orn_rand()
        if objects:
            self.pick_rand_objects()
        if visuals:
            self.visuals_rand()
        if objects_pos_orn:
            if falling:
                self.objects_pos_orn_rand_falling()
            else:
                self.objects_pos_orn_rand()
        if camera:
            self.camera_rand()