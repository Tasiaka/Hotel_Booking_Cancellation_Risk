"""Sanity-check model used only to validate the dataset formation logic, not as the final modeling homework."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


def topk_metrics(y_true, score, k=0.2):
    n = max(1, int(len(y_true) * k))
    order = np.argsort(score)[::-1][:n]
    y = np.asarray(y_true)
    precision = float(y[order].mean())
    recall = float(y[order].sum() / y.sum())
    lift = precision / float(y.mean())
    return precision, recall, lift


def run(path='data/processed/main_modeling_dataset.csv'):
    df = pd.read_csv(path)
    num_cols = ['lead_time','arrival_date_year','arrival_month_num','arrival_date_week_number','arrival_date_day_of_month',
                'stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','is_repeated_guest',
                'previous_cancellations','previous_bookings_not_canceled','adr','required_car_parking_spaces',
                'total_of_special_requests','total_nights','total_guests','has_children','has_previous_cancellations',
                'has_special_requests','is_long_lead_booking','is_weekend_stay','has_company','country_missing','agent_missing']
    cat_cols = ['hotel','arrival_date_month','meal','market_segment','distribution_channel','reserved_room_type','deposit_type','customer_type']
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ])
    model = Pipeline([('pre', pre), ('model', LogisticRegression(max_iter=300, class_weight='balanced', solver='liblinear'))])
    model.fit(train[num_cols + cat_cols], train['is_canceled'])
    score = model.predict_proba(test[num_cols + cat_cols])[:, 1]
    p20, r20, lift20 = topk_metrics(test['is_canceled'], score, 0.2)
    print({
        'roc_auc': roc_auc_score(test['is_canceled'], score),
        'pr_auc': average_precision_score(test['is_canceled'], score),
        'precision20': p20,
        'recall20': r20,
        'lift20': lift20,
    })


if __name__ == '__main__':
    run()


