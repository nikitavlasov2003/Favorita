# src/data.py
"""
Модуль для загрузки и базовой предобработки данных Favorita
"""

import polars as pl
import os
from typing import Dict

from .config import (
    TRAIN_PATH, TEST_PATH, STORES_PATH, ITEMS_PATH, OIL_PATH, HOLIDAYS_PATH
)


def load_clean_merged_df(base_path):
    stores = pl.read_csv(os.path.join(base_path, "stores.csv")).select([
        pl.col("store_nbr").cast(pl.Int8),
        pl.col("city").cast(pl.Categorical),
        pl.col("state").cast(pl.Categorical),
        pl.col("type").alias("store_type").cast(pl.Categorical),
        pl.col("cluster").cast(pl.Int8)
    ])
    
    items = pl.read_csv(os.path.join(base_path, "items.csv")).select([
        pl.col("item_nbr").cast(pl.Int32),
        pl.col("family").cast(pl.Categorical),
        pl.col("class").cast(pl.Int16),
        pl.col("perishable").cast(pl.Int8)
    ])
    
    oil = pl.read_csv(os.path.join(base_path, "oil.csv"), try_parse_dates=True).cast({
        "dcoilwtico": pl.Float32
    })
    
    holidays = (
        pl.read_csv(os.path.join(base_path, "holidays_events.csv"), try_parse_dates=True)
        .unique(subset=["date"])
        .rename({"type": "holiday_type"})
    )

    # Загрузка основного train (2017 год) - будь брать именно данный год, чтобы избежать возможных неприятностей с аномалией 2016 года
    train_2017 = (
        pl.scan_csv(os.path.join(base_path, "train.csv"), try_parse_dates=True)
        .filter(pl.col("date") >= pl.date(2017, 1, 1))
        .with_columns([
            pl.col("unit_sales").clip(lower_bound=0).cast(pl.Float32),
            pl.col("store_nbr").cast(pl.Int8),
            pl.col("item_nbr").cast(pl.Int32),
            # Явная логика преобразования строки в число 0/1
            pl.when(pl.col("onpromotion") == "True")
              .then(1)
              .otherwise(0)
              .cast(pl.Int8)
              .alias("onpromotion")
        ])
        .collect()
    )

    df = (
        train_2017
        .join(items, on="item_nbr", how="left")
        .join(stores, on="store_nbr", how="left")
        .join(oil, on="date", how="left")
        .join(holidays, on="date", how="left")
    )

    return df, stores, items, oil, holidays


def prepare_ml_features(train_df, items_df, stores_df, oil_df, holidays_df, is_test=False):
    # LazyFrame для оптимизации
    ldf = train_df.lazy()
    
    # Логирование таргета
    if not is_test:
        ldf = ldf.with_columns(
            pl.col('unit_sales').clip(lower_bound=0).log1p().alias('log_unit_sales')
        )

    # Присоединяем справочники
    ldf = ldf.join(items_df.lazy(), on='item_nbr', how='left')
    ldf = ldf.join(stores_df.lazy(), on='store_nbr', how='left')

    # Праздники (агрегация до 1 флага на дату)
    holidays_clean = (
        holidays_df.lazy()
        .filter(pl.col('transferred') == False)
        .group_by('date')
        .agg(pl.lit(1).alias('is_holiday').cast(pl.Int8))
    )
    ldf = ldf.join(holidays_clean, on='date', how='left').with_columns(
        pl.col('is_holiday').fill_null(0)
    )

    # Нефть и лаги по нефти
    oil_features = (
        oil_df.lazy()
        .with_columns(pl.col('dcoilwtico').fill_null(strategy='forward'))
        .with_columns([
            pl.col('dcoilwtico').shift(1).alias('oil_lag1'),
            pl.col('dcoilwtico').rolling_mean(window_size=7).alias('oil_roll_mean7')
        ])
    )
    ldf = ldf.join(oil_features, on='date', how='left')

    # Временные признаки (с твоим исправлением payroll)
    ldf = ldf.with_columns([
        pl.col('date').dt.month().alias('month').cast(pl.Int8),
        pl.col('date').dt.weekday().alias('day_of_week').cast(pl.Int8),
        pl.col('date').dt.day().alias('day_of_month').cast(pl.Int8),
        # Выходные
        (pl.col('date').dt.weekday() >= 6).cast(pl.Int8).alias('is_weekend'),
        ((pl.col('date').dt.day() == 15) | (pl.col('date') == pl.col('date').dt.month_end()))
            .cast(pl.Int8).alias('is_payroll'),
        # Декабрь
        (pl.col('date').dt.month() == 12).cast(pl.Int8).alias('is_december')
    ])

    # Лаги и скользящие (Window functions)
    ldf = ldf.sort(['store_nbr', 'item_nbr', 'date'])
    
    ldf = ldf.with_columns([
        # Лаги продаж (за 16 дней - начало теста)
        pl.col('log_unit_sales').shift(16).over(['store_nbr', 'item_nbr']).alias('lag_16'),
        pl.col('log_unit_sales').shift(21).over(['store_nbr', 'item_nbr']).alias('lag_21'),
        # Скользящее среднее (тренд)
        pl.col('log_unit_sales').shift(16)
            .rolling_mean(window_size=28)
            .over(['store_nbr', 'item_nbr'])
            .alias('rolling_mean_28'),
        # Лаг по промо
        pl.col('onpromotion').shift(16).over(['store_nbr', 'item_nbr']).alias('promo_lag_16')
    ])

    # Чистка Null и оптимизация типов
    # Оставляем категориальные признаки как есть, кастим только числа
    return ldf.drop_nulls(subset=['lag_16']).with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32)
    ])
    

