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


def load_metadata() -> Dict[str, pl.DataFrame]:
    """
    Загружает маленькие справочники целиком (stores, items, oil, holidays)
    Возвращает словарь с уже собранными DataFrame
    """
    metadata = {}
    
    # stores
    metadata['stores'] = pl.read_csv(STORES_PATH).with_columns([
        pl.col('store_nbr').cast(pl.Int8),
        pl.col('cluster').cast(pl.Int8),
        pl.col('type').cast(pl.Categorical)
    ])
    
    # items
    metadata['items'] = pl.read_csv(ITEMS_PATH).with_columns([
        pl.col('item_nbr').cast(pl.Int32),
        pl.col('class').cast(pl.Int16),
        pl.col('perishable').cast(pl.Int8),
        pl.col('family').cast(pl.Categorical)
    ])
    
    # oil
    metadata['oil'] = pl.read_csv(OIL_PATH, try_parse_dates=True).with_columns([
        pl.col('dcoilwtico').cast(pl.Float32)
    ])
    
    # holidays
    metadata['holidays'] = pl.read_csv(HOLIDAYS_PATH, try_parse_dates=True).with_columns([
        pl.col('transferred').cast(pl.Boolean),
        pl.col('type').cast(pl.Categorical),
        pl.col('locale').cast(pl.Categorical)
    ])
    
    return metadata


def prepare_data(
    df: pl.LazyFrame,
    items_df: pl.DataFrame,
    stores_df: pl.DataFrame,
    holidays_df: pl.DataFrame,
    oil_df: pl.DataFrame,
    is_test: bool = False
) -> pl.LazyFrame:
    """
    Базовая подготовка данных: join справочников, фичи праздников/нефти, временные признаки
    Возвращает LazyFrame — дальше можно добавлять лаги/rolling без collect
    """
    ldf = df.lazy()

    # 1. Логируем продажи (только train)
    if not is_test:
        ldf = ldf.with_columns(
            pl.col('unit_sales').clip(lower_bound=0).log1p().alias('log_unit_sales')
        )

    # 2. Join items (perishable + family)
    ldf = ldf.join(
        items_df.lazy().select('item_nbr', 'perishable', 'family'),
        on='item_nbr',
        how='left'
    )

    # 3. Join stores
    ldf = ldf.join(stores_df.lazy(), on='store_nbr', how='left')

    # 4. Праздники — is_holiday = True, если не transferred
    holidays_prep = (
        holidays_df.lazy()
        .with_columns(
            (~pl.col('transferred').fill_null(False)).alias('is_holiday')
        )
        .group_by('date')
        .agg(pl.max('is_holiday').alias('is_holiday'))
    )
    ldf = ldf.join(holidays_prep, on='date', how='left')

    # 5. Нефть — forward fill + лаги (без заглядывания в будущее!)
    oil_prep = (
        oil_df.lazy()
        .with_columns(pl.col('dcoilwtico').fill_null(strategy='forward'))
        .with_columns([
            pl.col('dcoilwtico').shift(1).alias('dcoilwtico_lag1'),
            pl.col('dcoilwtico').shift(7).alias('dcoilwtico_lag7'),
            pl.col('dcoilwtico').shift(14).alias('dcoilwtico_lag14'),
            pl.col('dcoilwtico').shift(28).alias('dcoilwtico_lag28')
        ])
    )
    ldf = ldf.join(
        oil_prep.select([
            'date', 'dcoilwtico', 'dcoilwtico_lag1', 'dcoilwtico_lag7',
            'dcoilwtico_lag14', 'dcoilwtico_lag28'
        ]),
        on='date',
        how='left'
    )

    # 6. Заполняем onpromotion
    ldf = ldf.with_columns(pl.col('onpromotion').fill_null(False))

    # 7. Временные признаки (weekday начинается с 1=Monday)
    ldf = ldf.with_columns([
        pl.col('date').dt.year().alias('year'),
        pl.col('date').dt.month().alias('month'),
        pl.col('date').dt.day().alias('day_of_month'),
        pl.col('date').dt.weekday().alias('day_of_week'),
        pl.col('date').dt.week().alias('week_of_year'),
        pl.col('date').dt.ordinal_day().alias('day_of_year'),
        pl.col('date').dt.weekday().is_in([6, 7]).alias('is_weekend')  # 6=Sat, 7=Sun
    ])

    return ldf


def load_train_chunk(days_back: int = 365) -> pl.LazyFrame:
    """
    Лениво загружает train за последние N дней (экономит память)
    """
    cutoff = pl.date(2017, 8, 15) - pl.duration(days=days_back)
    return (
        pl.scan_csv(TRAIN_PATH)
        .filter(pl.col('date') >= cutoff)
        .with_columns(pl.col('date').cast(pl.Date))
        .with_columns(pl.col('onpromotion').cast(pl.Boolean))
        .with_columns(pl.col('unit_sales').cast(pl.Float32))
        .with_columns(pl.col('store_nbr').cast(pl.Int8))
        .with_columns(pl.col('item_nbr').cast(pl.Int32))
    )

