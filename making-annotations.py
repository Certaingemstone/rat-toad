# script to create annotations from image folders

import os
import pandas as pd

rat_fns = os.listdir("Rats")
toad_fns = os.listdir("Frogs")
annotations = [0] * len(rat_fns) + [1] * len(toad_fns)
all_fns = rat_fns + toad_fns

df = pd.DataFrame()
df["names"] = all_fns
df["labels"] = annotations

df.to_csv("labels.csv", index=False)