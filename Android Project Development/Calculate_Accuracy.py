import numpy as np
from sklearn import metrics


def main(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred, multioutput='raw_values')
    mae = metrics.mean_absolute_error(y_true, y_pred, multioutput='raw_values') 
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = metrics.r2_score(y_true, y_pred, multioutput='raw_values')
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    count = len(y_true)
    exp_var = metrics.explained_variance_score(y_true, y_pred, multioutput='raw_values')
    
    
    return mse, mae, r, r2, mape, count, exp_var