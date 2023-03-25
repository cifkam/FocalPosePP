import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from focalpose.simulator import Camera
from focalpose.recording.bop_recording_scene import BopRecordingScene, SamplerError
from focalpose.config import LOCAL_DATA_DIR
from focalpose.datasets.real_dataset import Pix3DDataset, CompCars3DDataset, StanfordCars3DDataset
from focalpose.fitting.nonparametric_model import NonparametricModel
from focalpose.fitting.fitting import get_outliers

TOP,LEFT,BOTTOM,RIGHT=range(4)

class BopRecordingSceneNonparametric(BopRecordingScene):
    def __init__(self,
                 deltas=None,
                 outliers = 0.05,
                 nonparam_q=0.95,
                 soft_border_check_enlargement=2,
                 soft_border_check_treshold=0.50,
                 area_check=0.03,

                 urdf_ds='ycbv',
                 texture_ds='shapenet',
                 domain_randomization=True,
                 background_textures=False,
                 textures_on_objects=False,
                 n_objects_interval=(1, 1),
                 #objects_xyz_interval=((0.0, -0.5, -0.15), (1.0, 0.5, 0.15)),
                 proba_falling=0.0,
                 resolution=(640, 480),
                 #focal_interval=(515, 515),
                 #camera_distance_interval=(0.5, 1.5),
                 border_check=False,
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

        assert (not soft_border_check_enlargement and not soft_border_check_treshold) or \
            (soft_border_check_enlargement > 1 and soft_border_check_treshold >= 0 and soft_border_check_treshold <= 1)

        self.soft_border_check_enlargement = soft_border_check_enlargement
        self.soft_border_check_treshold = soft_border_check_treshold
        self.area_check=area_check
        
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
            self.nonparametric_model = NonparametricModel.fit(self.real_dataset, nonparam_q)
        else:
            self.nonparametric_model = NonparametricModel(
                self.real_dataset,
                deltas['R'],
                deltas['x'],
                deltas['y'],
                deltas['z'],
                deltas['f'])

    def sample_camera(self):
        TWC,f = self.sample_TWC_f()
        return self.create_camera(TWC,f)

    def sample_TWC_f(self):
        R,t,f = self.nonparametric_model.sample()
        Rt = np.hstack([R,t.reshape(-1,1)])
        TWC = np.vstack([ Rt , [0,0,0,1] ])
        return TWC,f

    def create_camera(self, TWC, f, enlargement_factor=1):
        K = np.zeros((3, 3), dtype=np.float)
        W, H = max(self.resolution), min(self.resolution)
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = W / 2 * enlargement_factor
        K[1, 2] = H / 2 * enlargement_factor
        K[2, 2] = 1.0
        cam = Camera(resolution=self.resolution, client_id=self._client_id)
        h,w = self.resolution
        cam.set_intrinsic_K(K, h=h*enlargement_factor, w=w*enlargement_factor)
        cam.set_extrinsic_T(TWC)
        return cam

    @staticmethod
    def check_area(uniqs, cam_obs, q):
        mask = cam_obs['mask']
        for uniq in uniqs[uniqs > 0]:
            ids = np.where(mask == uniq)
            bbox_area = (ids[0].max()-ids[0].min()) * (ids[1].max()-ids[1].min())
            if bbox_area / (mask.shape[0]*mask.shape[1]) < q:
                return False
        return True

    @staticmethod
    def intersection_area(a, b):
        intersection = (
            max(a[TOP],    b[TOP]),
            max(a[LEFT],   b[LEFT]),
            min(a[BOTTOM], b[BOTTOM]),
            min(a[RIGHT],  b[RIGHT]))

        if intersection[LEFT] < intersection[RIGHT] and intersection[TOP] < intersection[BOTTOM]:
            return (intersection[BOTTOM]-intersection[TOP])*(intersection[RIGHT]-intersection[LEFT])
        else:
            return 0

    @staticmethod
    def check_border_soft(uniqs, cam_obs, resolution, q, treshold):
        mask = cam_obs['mask']
        h,w = resolution
        img_h = h/q
        img_w = w/q
        padding_h = (h - img_h)/2
        padding_w = (w - img_w)/2
        img_bbox  = (padding_h, padding_w, padding_h+img_h, padding_w+img_w)

        for uniq in uniqs[uniqs > 0]:
            ids = np.where(mask == uniq)
            object_bbox = (ids[0].min(), ids[1].min(), ids[0].max(), ids[1].max())
            object_area = (object_bbox[BOTTOM]-object_bbox[TOP])*(object_bbox[RIGHT]-object_bbox[LEFT])
            intersection_area = BopRecordingSceneNonparametric.intersection_area(img_bbox, object_bbox)

            if intersection_area / object_area < treshold:
                return False
        
        return True


    def camera_rand(self):
        N = 0
        valid = False
        self.cam_obs = None

        while not valid:
            N += 1
            if N > 3:
                raise SamplerError('Cannot sample valid camera configuration.')
            
            TWC,f = self.sample_TWC_f()
            cam = self.create_camera(TWC, f, enlargement_factor=self.soft_border_check_enlargement)
            cam_obs_ = cam.get_state()
            mask = cam_obs_['mask']
            mask[mask == self.background._body_id] = 0
            mask[mask == 255] = 0
            uniqs = np.unique(cam_obs_['mask'])

            
            valid = len(uniqs) == len(self.bodies) + 1 and np.sum(mask) > 0
            if not valid: continue
            
            if self.soft_border_check_enlargement and not self.border_check:
                # check that object is inside enlarged image and that image contains big enough portion of object's bbox
                valid = self.check_border(uniqs, cam_obs_) and self.check_border_soft(uniqs, cam_obs_, 
                    self.resolution,
                    self.soft_border_check_enlargement,
                    self.soft_border_check_treshold)
                if not valid: continue
        
            if self.soft_border_check_enlargement != 1:
                cam = self.create_camera(TWC, f, enlargement_factor=1)
                cam_obs_ = cam.get_state()
                mask = cam_obs_['mask']
                mask[mask == self.background._body_id] = 0
                mask[mask == 255] = 0
                uniqs = np.unique(cam_obs_['mask'])
                valid = len(uniqs) == len(self.bodies) + 1 and np.sum(mask) > 0
                if not valid: continue

            if self.border_check and not self.soft_border_check_enlargement:
                valid = self.check_border(uniqs, cam_obs_)
                if not valid: continue

            if self.area_check:
                valid = self.check_area(uniqs, cam_obs_, self.area_check)
                
        self.cam_obs = cam_obs_
    
            
        
    def objects_pos_orn_rand(self):
        self.hide_plane()
        for body in self.bodies:
            pos = np.zeros(3)
            orn = pin.Quaternion().coeffs()
            body.pose = pos, orn
