from focalpose.config import SYNT_DS_DIR
from pathlib import Path
from focalpose.datasets.synthetic_dataset import SyntheticSceneDataset
import numpy as np
import numpy.linalg
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', default=None, type=str)
parser.add_argument('--overwrite', action='store_true', default=False)

def prepare(ds_dir):
    filename = ds_dir / 'camera.pkl'
    if not args.overwrite and filename.is_file():
        return False

    try:
        ds = SyntheticSceneDataset(ds_dir)
    except:
        return False

    print(f"Working on {ds_dir}")
    rows = []

    for i in tqdm(range(len(ds))):
        cam = ds[i][2]['camera']
        TCO = np.linalg.inv(cam['TWC'])
        K = cam['K']
        rows.append({'K':K, 'TCO':TCO})
        #rows.append(ds[i][2])

    df = pd.DataFrame(rows)
    df.to_pickle(filename)
    return True


if __name__ == '__main__':
    args = parser.parse_args([] if '__file__' not in globals() else None)

    if args.dataset != None:
        prepare(SYNT_DS_DIR / args.dataset)

    else:
        for ds_dir in SYNT_DS_DIR.iterdir():
            prepare(ds_dir)

        dirs = list(SYNT_DS_DIR.iterdir())
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
            all.to_pickle(SYNT_DS_DIR / ('camera-'+prefix+'.pkl'))

