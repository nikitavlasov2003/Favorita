last_week_data = train_subset.filter(
    pl.col('date') >= (val_start_date - datetime.timedelta(days=28))
).with_columns(
    pl.col('date').dt.weekday().alias('day_of_week')
)

# ❗ ФИКС — агрегация
last_week_data = last_week_data.group_by(
    ['store_nbr', 'item_nbr', 'day_of_week']
).agg(
    pl.col('unit_sales').mean().alias('pred_sales')
)

# 2. Join
snaive_pred = valid_subset.join(
    last_week_data,
    on=['store_nbr', 'item_nbr', 'day_of_week'],
    how='left'
)

# 3. Clean
snaive_pred = snaive_pred.with_columns([
    pl.col('unit_sales').clip(lower_bound=0).fill_null(0),
    pl.col('pred_sales').clip(lower_bound=0).fill_null(0)
])

# 4. Weights
weights = (
    snaive_pred['perishable']
    .replace({0: 1.0, 1: 1.25})
    .fill_null(1.0)
    .to_numpy()
)

# 5. Log
y_true_log = np.log1p(snaive_pred['unit_sales'].to_numpy())
y_pred_log = np.log1p(snaive_pred['pred_sales'].to_numpy())

baseline_score = nwrmsle(y_true_log, y_pred_log, weights)
print(f"Seasonal Naive NWRMSLE: {baseline_score:.4f}")