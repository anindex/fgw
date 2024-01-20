import pandas as pd 
import os

## add mol_id
sets = ["train", "test", "val"]
_DIR = "/home/taing/workspace/2d_3d_molecular/data/cov_2_gen"
for s in sets:
    df = pd.read_csv(os.path.join(_DIR, s+".csv"))
    df["mol_id"] = range(len(df))
    df.to_csv(os.path.join(_DIR, s+".csv"))
