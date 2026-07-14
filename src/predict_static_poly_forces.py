"""
Created on Fri Dec 30 10:25:14 2022

@author: Malika In-Albon
"""

import pickle
from sklearn.metrics import (mean_squared_error, mean_absolute_error)

from load_data import LoadData
from feature_engineering_utils import *

# List of folders in which the data are stored
list_folder = ['amp_0.2_freq_0.1', 'amp_0.2_freq_0.5', 'amp_0.2_freq_1.0',
               'amp_0.35_freq_0.1', 'amp_0.35_freq_0.5', 'amp_0.35_freq_1.0',
               'amp_0.5_freq_0.1', 'amp_0.5_freq_0.5', 'amp_0.5_freq_1.0']

# Initialize lists of metrics
list_r2 = []
list_mse = []
list_rmse = []
list_mae = []
list_metrics_name = ['MSE', 'RMSE', 'MAE']

# ------------------------- Load data ------------------------------------
# Static polymander for prediction
poly_static_FL = LoadData()
for folder in list_folder:
    poly_static_FL.load_polymander_data(dir_name=f'logs_polymander/static/FL/prediction/{folder}')
    poly_static_FL.load_force_plates_data(dir_name=f'logs_force_plates/static/FL/prediction/{folder}')
print(f'{len(poly_static_FL.list_polymander)} files in list_polymander')
print(f'{len(poly_static_FL.list_force_plates)} files in list_force_plate')

# ------------------------------- Load model-------------------------
mlr = pickle.load(open('models/mlr.pkl', 'rb'))

# -------------------------------- Feature engineering ----------------------------
check = False
for (i, j) in zip(poly_static_FL.list_polymander, poly_static_FL.list_force_plates):
    # Signal processing
    t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = prepare_data_for_training(i, j)

    # -------------------------- Prediction of Fz for FL limb of walking polymander ---------------
    X_test = fbck_current_final[:, 8:10]
    y_test = Fxyz_final[:, 2]
    y_pred = mlr.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print('\n----------------------------- metrics -------------------------------')
    print(f'mlr metrics : [MSE, RMSE, MAE] = [{mse:.2f}, {rmse:.2f}, {mae:.2f}]')

    list_mse.append(mse)
    list_rmse.append(rmse)
    list_mae.append(mae)

    # Plot Fz as a function of hip and calf current
    plot_3d_currents_force(y_pred, fbck_current_final[:, 8], fbck_current_final[:, 9])

    # Plot prediction of Fz
    fig, ax = plt.subplots()
    ax.set_title('Prediction of Fz with hip and calf motors')
    ax.plot(t_s_final, y_test, label='true value')
    ax.plot(t_s_final, y_pred, label='pred')
    ax.set(xlabel='time [s]', ylabel='Fz [N]')
    ax.legend()

list_metrics = [list_mse, list_rmse, list_mae]

# Plot 3D bar chart of error depending on amplitude and frequency
for (metric, metric_name) in zip(list_metrics, list_metrics_name):
    plot_3d_metrics(metric, metric_name)
    plt.savefig(f'figures/lr_results/3D_bar_chart_robust_model_{metric_name}.png', format='png')

plt.show()

