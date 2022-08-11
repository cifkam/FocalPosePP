import numpy as np
from scipy.spatial.transform import Rotation

class NonparametricModel():
    def __init__(self, real_dataset, delta_R, delta_x, delta_y, delta_z, delta_f):
        self.real_dataset = real_dataset
        self.delta_R = delta_R
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_z = delta_z
        self.delta_f = delta_f

    @staticmethod
    def fit(real_dataset, q=98):
        return NonparametricModel(real_dataset, *NonparametricModel.fit_params(real_dataset, q))
    
    @staticmethod
    def fit_params(real_dataset, q=98):
        xy = real_dataset.TWC[:,:2,3]
        zf = np.vstack([real_dataset.TWC[:,2,3], real_dataset.f]).T
        R = real_dataset.TWC[:,:3,:3]
        delta_x, delta_y = NonparametricModel.get_delta(xy, q)
        delta_z, delta_f = NonparametricModel.get_delta(zf, q)
        delta_R = NonparametricModel.get_delta_rot(R, q)
        return delta_R, delta_x, delta_y, delta_z, delta_f

    def sample(self):
        i = np.random.randint(0, len(self.real_dataset))
        dx,dy = NonparametricModel.sample_from_unit_sphere(2) * np.array([self.delta_x,self.delta_y])
        dz,df = NonparametricModel.sample_from_unit_sphere(2) * np.array([self.delta_z,self.delta_f])
        dR = Rotation.from_rotvec(NonparametricModel.sample_from_unit_sphere(3) * self.delta_R).as_matrix()

        f = self.real_dataset.f[i] + df
        TWC = self.real_dataset.TWC[i]
        x = TWC[0,3] + dx
        y = TWC[1,3] + dy
        z = TWC[2,3] + dz
        R = TWC[:3,:3] @ dR

        return R, np.array([x,y,z]), f
    
    def sample_n(self, n):
        dR = NonparametricModel.sample_from_unit_sphere(3,n) * self.delta_R
        dR = np.array(list(map(lambda x: Rotation.from_rotvec(x).as_matrix(), dR)))
        dxy = NonparametricModel.sample_from_unit_sphere(2,n) * np.array([self.delta_x, self.delta_y])
        dzf = (NonparametricModel.sample_from_unit_sphere(2,n) * np.array([self.delta_z, self.delta_f]) )

        dt = np.hstack([dxy, dzf[:,0].reshape(-1,1)])
        df = dzf[:,1]

        indices = np.random.choice(n, size=n)
        TWC = self.real_dataset.TWC[indices]
        R = TWC[:,:3,:3] @ dR
        t = TWC[:,:3,3] + dt
        f = self.real_dataset.f[indices] + df

        return R,t,f



    @staticmethod
    def nearest_dists(data):
        dists = []
        for i in range(data.shape[0]):
            if len(data.shape) == 1:
                dist2 = (data[i]-data)**2
            else:
                dist2 = np.sum((data[i]-data)**2, axis=-1)
            dists.append(np.sqrt(dist2[np.argpartition(dist2, 1)[1]]))
        return dists

    @staticmethod
    def nearest_dists_rot(data):
        dists = []
        for i in range(data.shape[0]):
            R_deltas = data[i].T @ data
            dist2 = np.arccos( np.clip((np.trace(R_deltas, axis1=1, axis2=2)-1)/2, -1, 1) )
            dists.append( dist2[np.argpartition(dist2, 1)[1]] )
        return dists

    @staticmethod
    def get_delta(data, q):
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        ranges = maxs - mins
        ranges[ranges==0] = 1
        data_norm = (data - mins) / ranges
        delta = np.percentile(NonparametricModel.nearest_dists(data_norm), q) 
        return delta * ranges

    @staticmethod
    def get_delta_rot(data, q):
        return np.percentile(NonparametricModel.nearest_dists_rot(data), q)

    @staticmethod
    def sample_from_unit_sphere(dim, samples=1):
        rng = np.random.default_rng()
        if samples == 1:
            X = rng.normal(size=(dim))
            U = rng.random(1) 
            return U**(1/dim) / np.sqrt(np.sum(X**2, keepdims=True)) * X
        else:
            X = rng.normal(size=(samples , dim))
            U = rng.random((samples, 1)) 
            return U**(1/dim) / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X
