"""
Metrics for Favorita forecasting.
"""
import numpy as np

def nwrmsle(y_true, y_pred, weights):
    # Метрика: Root Mean Squared Logarithmic Error с весами
    # y_true и y_pred уже должны быть в логарифмической шкале, 
    # если мы обучаем модель на log(1 + unit_sales)
    
    # Но если они в обычных единицах:
    # res = np.square(np.log1p(y_pred) - np.log1p(y_true))
    
    # Если мы уже работаем с логарифмами (что эффективнее):
    res = np.square(y_pred - y_true)
    weighted_res = res * weights
    return np.sqrt(np.sum(weighted_res) / np.sum(weights))


