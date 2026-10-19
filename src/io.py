import h5py
import pandas as pd


def list_keys(path):
    with h5py.File(path, "r") as f:
        return list(f.keys())


def read_matrix(path, key, index_key=None):
    with h5py.File(path, "r") as f:
        values = f[key][:]
        if index_key and index_key in f:
            index = [x.decode() if isinstance(x, bytes) else x for x in f[index_key][:]]
        else:
            index = [str(i) for i in range(values.shape[0])]
    return pd.DataFrame(values, index=index)


def load_omics(path, spec):
    return {name: read_matrix(path, key, idx) for name, (key, idx) in spec.items()}
