"""
Created on Mon Dec 26 09:17:03 2022

@author: Malika In-Albon
"""

import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

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
list_metrics_name = ['R2', 'MSE', 'RMSE', 'MAE']

# Train 9 models by modulating amplitude and frequency
for folder in list_folder:
    # -------------------------------- Load data ----------------------------------------
    load_data = LoadData()
    load_data.load_polymander_data(dir_name=f'logs_polymander/static/FL/{folder}')
    load_data.load_force_plates_data(dir_name=f'logs_force_plates/static/FL/{folder}')
    print(f'{len(load_data.list_polymander)} files in list_polymander')
    print(f'{len(load_data.list_force_plates)} files in list_force_plate')

    # -------------------------------- Feature engineering ----------------------------
    check = False
    for (i, j) in zip(load_data.list_polymander, load_data.list_force_plates):
        # Signal processing
        t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = prepare_data_for_training(i, j)

        # Store data in X and y
        if check is False:
            X = fbck_current_final[:, 8:10]
            y = Fxyz_final[:, 2]
            check = True
        elif check is True:
            X = np.concatenate((X, fbck_current_final[:, 8:10]))
            y = np.concatenate((y, Fxyz_final[:, 2]))

    # ------------------------ Supervised learning ----------------------------------
    # Split the data in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

    # Create and train model - multiple linear regression
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)

    # Save model
    pickle.dump(mlr, open(f'models/mlr_{folder}.pkl', 'wb'))
    y_pred = mlr.predict(X_test)

    # Metrics
    # r2 = mlr.score(X_train, y_train)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_test, y_pred)
    # print('\n----------------------------- metrics -------------------------------')
    # print(f'mlr metrics : [R^2, MSE, RMSE, MAE] = [{r2:.2f}, {mse:.2f}, {rmse:.2f}, {mae:.2f}]')

    # K-fold cross validation
    folds = KFold(n_splits=5, shuffle=True, random_state=0)

    r2_scores = cross_val_score(mlr, X_train, y_train, scoring='r2', cv=folds)
    mse_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_mean_squared_error', cv=folds)
    rmse_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_root_mean_squared_error', cv=folds)
    mae_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_mean_absolute_error', cv=folds)

    mse_scores = abs(mse_scores)
    rmse_scores = abs(rmse_scores)
    mae_scores = abs(mae_scores)

    print('\n------------------------ k-fold cross validation----------------------')
    print("R2: mean score of %0.4f with a standard deviation of %0.4f" % (r2_scores.mean(), r2_scores.std()))
    print("MSE: mean score of %0.4f with a standard deviation of %0.4f" % (mse_scores.mean(), mse_scores.std()))
    print("RMSE: mean score of %0.4f with a standard deviation of %0.4f" % (rmse_scores.mean(), rmse_scores.std()))
    print("MAE: mean score of %0.4f with a standard deviation of %0.4f\n" % (mae_scores.mean(), mae_scores.std()))

    # Store metric scores for every amplitude and frequency in a list
    list_r2.append(r2_scores)
    list_mse.append(mse_scores)
    list_rmse.append(rmse_scores)
    list_mae.append(mae_scores)

list_metrics = [list_r2, list_mse, list_rmse, list_mae]

# Plot 3D bar charts
for (metric_name, metric) in zip(list_metrics_name, list_metrics):

    metric = np.array(metric)
    results = np.mean(metric, 1)
    errors = abs(np.max(metric, 1) - np.min(metric, 1))

    plot_3d_metrics(results, metric_name, errors)

    plt.savefig(f'figures/lr_results/3D_bar_chart_{metric_name}.png', format='png')
    plt.savefig(f'figures/lr_results/3D_bar_chart_{metric_name}.eps', format='eps')

plt.show()
