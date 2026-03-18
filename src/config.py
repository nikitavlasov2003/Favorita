# src/config.py
import os
from datetime import date

DATA_DIR = "/kaggle/input/datasets/nikolasking/favorita"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
STORES_PATH = os.path.join(DATA_DIR, "stores.csv")
ITEMS_PATH = os.path.join(DATA_DIR, "items.csv")
OIL_PATH = os.path.join(DATA_DIR, "oil.csv")
HOLIDAYS_PATH = os.path.join(DATA_DIR, "holidays_events.csv")

VAL_START = date(2017, 8, 1)
VAL_END = date(2017, 8, 15)

FORECAST_HORIZON = 16
SEASONAL_WINDOW = 28