"""Evaluate a training-mean predictor on the submitted notebook's exact split."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=root/"HotelReviewDataset.csv")
data_path=parser.parse_args().data
data=pd.read_csv(data_path)
y=data["Review_Score"]
rest,test=train_test_split(y,test_size=.15,random_state=42)
train,val=train_test_split(rest,test_size=.15/(.7+.15),random_state=42)
assert (len(train),len(val),len(test))==(9640,2066,2066)
assert np.allclose(test.iloc[:10],[7.1,6.3,5.8,5.8,6.3,6.3,4.2,6.7,6.7,2.5])
mean_mse=float(mean_squared_error(test,np.repeat(train.mean(),len(test))))
recorded_mse=.0331  # Rounded normalized MSE printed by the original notebook.
result={
    "baseline_provenance":"Training-mean baseline computed on 2026-11-15 using the original split.",
    "neural_provenance":"Saved June 2025 notebook output; not retrained in this update. MSE was printed to four decimal places.",
    "dataset_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "split":{"train":len(train),"validation":len(val),"test":len(test),"seed":42},
    "training_mean":float(train.mean()),
    "baseline":{"normalized_mse":mean_mse/100,"mse":mean_mse,"rmse":mean_mse**.5},
    "recorded_neural":{"normalized_mse":recorded_mse,"mse":recorded_mse*100,"rmse":(recorded_mse*100)**.5},
}
(root/"results").mkdir(exist_ok=True)
(root/"results/comparison.json").write_text(json.dumps(result,indent=2)+"\n")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig,ax=plt.subplots(figsize=(8.5,3.2),layout="constrained")
bars=ax.barh(["Training-mean baseline · rerun", "BiLSTM + metadata · recorded"],
    [result["baseline"]["rmse"],result["recorded_neural"]["rmse"]],color=["#a5b6c5","#317c9c"],height=.5)
ax.bar_label(bars,fmt="%.3f",padding=7)
ax.invert_yaxis();ax.set(xlim=(0,2.4),xlabel="Test RMSE · score points",title="Hotel reviews · same 2,066-row test split")
ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0,pad=8)
(root/"assets").mkdir(exist_ok=True)
fig.savefig(root/"assets/test-comparison.png",dpi=180,facecolor="white")
print(json.dumps(result,indent=2))
