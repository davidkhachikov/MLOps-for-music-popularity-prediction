from src.model import (load_features, log_metadata, train, \
                   retrieve_model_with_alias, retrieve_model_with_version)
import pandas as pd
from src.utils import init_hydra

def test_load_features():
    pass
    # cfg = init_hydra()
    # train_data_version = cfg.train_data_version
    # X, y = load_features(name = "features_target", version=train_data_version)
    
    # assert isinstance(X, pd.Dataframe)
    # assert isinstance(y, pd.Dataframe)


def test_train():
    pass
    # cfg = init_hydra()
    # train_data_version = cfg.train_data_version

    # X_train, y_train = load_features(name = "features_target", version=train_data_version)

    # gs = train(X_train, y_train, cfg=cfg)

def test_log_metadata():
    pass

def test_retrieve_model_with_alias():
    pass

def test_retrieve_model_with_version():
    pass