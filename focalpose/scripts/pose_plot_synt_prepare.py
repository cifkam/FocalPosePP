from focalpose.config import SYNT_DS_DIR
from pathlib import Path
from focalpose.datasets.synthetic_dataset import SyntheticSceneDataset
import numpy as np
import numpy.linalg
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    for ds_dir in Path(SYNT_DS_DIR).iterdir():
        filename = ds_dir / 'camera.pkl'
        if filename.is_file():
            continue

        print(f"Working on {ds_dir}")
        ds = SyntheticSceneDataset(ds_dir)
        rows = []

        for i in tqdm(range(len(ds))):
            cam = ds[i][2]['camera']
            TCO = np.linalg.inv(cam['TWC'])
            K = cam['K']
            rows.append({'K':K, 'TCO':TCO})
            #rows.append(ds[i][2])

        df = pd.DataFrame(rows)
        df.to_pickle(filename)
    


    dirs = list(Path(SYNT_DS_DIR).iterdir())
    ds_prefixes = [
        'compcars3d',
        'stanfordcars3d',
        'pix3d-chair',
        'pix3d-bed',
        'pix3d-sofa',
        'pix3d-table'
        ]

    for prefix in ds_prefixes:
        print(f"Merging {prefix}")
        dfs = []
        for dir in [x for x in dirs if x.parts[-1][:len(prefix)] == prefix]:
            dfs.append(pd.read_pickle(dir/'camera.pkl'))
        all = pd.concat(dfs)
        all.to_pickle(Path(SYNT_DS_DIR) / ('camera-'+prefix+'.pkl'))