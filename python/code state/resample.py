"""Collection of I/O utility functions for estimating Covid-19 models."""
import json
import numpy as np
from typing import List, Text, Union
Numeric = Union[int, float]

# 0. Convert training data into metadata.json
# 1. resample data from residual
# 2. Collect output from point estimation to save in residuals.json

def read_residuals(fpath: Text, n_states: int):
  with open(fpath, "r") as f:
    jdata = json.load(f)
  if len(jdata.keys()) != n_states:
    raise ValueError(f"The residual data should contains exactly {n_states} records.")
  for k, v in jdata.items():
    if set(v.keys()) != {"state", "fitted", "residual"}:
      raise ValueError(f"At the record {k}, the json data don't follow the required format, "
                       "expected to have three fields: state, fitted and residual.")
  return jdata


def get_resampled_input(fitted: List[List[Numeric]], residuals: List[List[Numeric]],
                        seed: int) -> List[List[Numeric]]:
    # raise NotImplemented
    len_train = len(residuals)
    index1 = np.random.permutation(range(34))
    index2 = np.random.permutation(range(34,len_train))
    index = np.concatenate([index1,index2])
    residual_perm = np.array(residuals)[index,:]
    fitted = np.array(fitted)
    perm_data = (residual_perm*np.sqrt(fitted[:len_train,:])+fitted[:len_train,:])
    perm_data = np.maximum(0,perm_data).tolist()

    return perm_data

