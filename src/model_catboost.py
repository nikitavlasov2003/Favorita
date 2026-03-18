from catboost import CatBoostRegressor
import gc
import numpy as np
import pandas as pd

def train_cb(train_df, val_df, features, cat_features):
    y_train = np.log1p(train_df['unit_sales'].clip(lower=0))
    y_val = np.log1p(val_df['unit_sales'].clip(lower=0))
    
    train_pool = cb.Pool(train_df[features], y_train, cat_features=cat_features)
    val_pool = cb.Pool(val_df[features], y_val, cat_features=cat_features)
    
    del y_train
    gc.collect()
    params = {
        'iterations': 2000,
        'learning_rate': 0.02,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',
        'early_stopping_rounds': 100,
        'verbose': 200
    }

    model = CatBoostRegressor(**params)

    # Обучение
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    return model