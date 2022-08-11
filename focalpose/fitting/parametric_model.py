from ast import Param
import numpy as np
from scipy.spatial.transform import Rotation
from deep_bingham.bingham_distribution import BinghamDistribution


class ParametricModel():
    def __init__(self, xy_mu, xy_cov, zf_log_mu, zf_log_cov, rot_bingham_m, rot_bingham_z):
        self.xy_mu = xy_mu
        self.xy_cov = xy_cov
        self.zf_log_mu = zf_log_mu
        self.zf_log_cov = zf_log_cov
        self.bingham = BinghamDistribution(np.array(rot_bingham_m), np.array(rot_bingham_z))

    @staticmethod
    def fit(real_dataset):
        return ParametricModel(*ParametricModel.fit_params(real_dataset))

    @staticmethod
    def fit_params(real_dataset):
        TCO = real_dataset.TCO
        f = real_dataset.f
        R = TCO[:,:3,:3]
        t = TCO[:,:3, 3]
        xy = t[:,:2]
        zf = np.vstack([t[:,2],f]).T

        xy_mu = np.mean(xy, axis=0)
        xy_cov = np.cov(xy.T)
        R_quat = np.array(list( map(lambda x: Rotation.from_matrix(x).as_quat(), R) ))
        bingham = BinghamDistribution.fit(R_quat)
        logzf = np.log(zf)
        zf_log_mu = np.mean(logzf, axis=0)
        zf_log_cov = np.cov(logzf.T)

        return xy_mu, xy_cov, zf_log_mu, zf_log_cov, bingham._param_m, bingham._param_z

    def sample(self):
        # Sample TCO
        x,y = np.random.multivariate_normal(self.xy_mu, self.xy_cov)
        z,f = np.exp( np.random.multivariate_normal(self.zf_log_mu, self.zf_log_cov) )

        q = self.bingham.random_samples(1)
        R = Rotation.from_quat(q).as_matrix().reshape((3,3))
        t = np.array([[x,y,z]]).T

        # Return TWC
        return R.T, -R.T@t, f

    def sample_n(self, n):
        # Sample TCO
        R = np.array(list(map(lambda x: Rotation.from_quat(x).as_matrix(), self.bingham.random_samples(n))))
        xy = np.random.multivariate_normal(self.xy_mu, self.xy_cov, size=n)
        zf = np.exp(np.random.multivariate_normal(self.zf_log_mu, self.zf_log_cov, size=n))
        z = zf[:,0]
        f = zf[:,1]
        t = np.hstack([xy,z.reshape(-1,1)])

        # Return TWC
        R_T = np.transpose(R,axes=(0,2,1))
        return R_T, (-R_T@t[:, :, None]).squeeze(), f
