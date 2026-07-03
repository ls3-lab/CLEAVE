import os
import pytest


@pytest.fixture(scope="session")
def cora_setup():
    """Cora + EPAGCL oracle-model retriever on CPU (deterministic float for exactness tests)."""
    from hydra import initialize_config_dir, compose
    from hydra.core.hydra_config import HydraConfig
    from src.data.loader import get_dataset
    from src.retrievers import get_retriever
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("conf")):
        cfg = compose(config_name="config", overrides=["dataset=cora", "retriever=epagcl"],
                      return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)   # resolves ${hydra:runtime.cwd} in retriever paths
    data = get_dataset(cfg.dataset)[0].to("cpu")
    retriever = get_retriever(cfg.retriever, cfg, data, "cpu", use_oracle=True)
    return cfg, data, retriever
