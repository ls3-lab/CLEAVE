import glob
import pickle
from build_oracle import atomic_pickle_dump


def test_atomic_dump_roundtrip(tmp_path):
    path = str(tmp_path / "cache.pkl")
    obj = {1: {"init_position": 3, "optimal": {1: {"edges": [(1, 2)]}}}}
    atomic_pickle_dump(obj, path)
    with open(path, "rb") as f:
        assert pickle.load(f) == obj


def test_atomic_dump_overwrites_and_leaves_no_tmp(tmp_path):
    path = str(tmp_path / "cache.pkl")
    atomic_pickle_dump({"a": 1}, path)
    atomic_pickle_dump({"a": 2}, path)
    with open(path, "rb") as f:
        assert pickle.load(f) == {"a": 2}
    assert glob.glob(str(tmp_path / "*.tmp")) == []
