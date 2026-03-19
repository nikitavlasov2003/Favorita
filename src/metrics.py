"""
Metrics for Favorita forecasting.
"""
import numpy as np

def nwrmsle(y_true, y_pred, weights):
    """
    y_true: фактические продажи (в исходной шкале!)
    y_pred: предсказанные продажи (в исходной шкале!)
    weights: массив весов (1.25 для perishable, 1.0 для остальных)
    """
    # Гарантируем отсутствие отрицательных значений
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    
    # Считаем взвешенную сумму квадратов разностей логарифмов
    weighted_sq_log_diff = weights * (np.log1p(y_pred) - np.log1p(y_true))**2
    
    return np.sqrt(np.sum(weighted_sq_log_diff) / np.sum(weights))


