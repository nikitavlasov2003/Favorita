# src/model_catboost.py
from catboost import CatBoostRegressor, Pool
import gc

def train_catboost(train_df, val_df, features, cat_features, **kwargs):

    # Convert polars to pandas if needed (CatBoost expects pandas/numpy)
    if hasattr(train_df, 'to_pandas'):
        train_df = train_df.to_pandas()
    if hasattr(val_df, 'to_pandas'):
        val_df = val_df.to_pandas()

    # Targets are already log-transformed
    y_train = train_df['log_unit_sales'].values
    y_val = val_df['log_unit_sales'].values

    train_pool = Pool(train_df[features], y_train, cat_features=cat_features)
    val_pool = Pool(val_df[features], y_val,   cat_features=cat_features)

    # Free some memory
    del y_train
    gc.collect()

    # Default parameters 
    params = {
        'iterations': 2000,
        'learning_rate': 0.02,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'task_type': 'GPU',
        'early_stopping_rounds': 100,
        'verbose': 200,
    }
    params.update(kwargs)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    return model
