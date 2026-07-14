"""
Created on Mon Jan 02 10:17:45 2022

@author: Malika In-Albon
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from load_data import LoadData
from feature_engineering_utils import *


# List of folder
list_amp_0 = ['amp_0.2_freq_0.1', 'amp_0.2_freq_0.5', 'amp_0.2_freq_1.0']
list_amp_1 = ['amp_0.35_freq_0.1', 'amp_0.35_freq_0.5', 'amp_0.35_freq_1.0']
list_amp_2 = ['amp_0.5_freq_0.1', 'amp_0.5_freq_0.5', 'amp_0.5_freq_1.0']
list_list_amp = [list_amp_0, list_amp_1, list_amp_2]

# List of metrics
list_r2 = []
list_mse = []
list_rmse = []
list_mae = []
list_metrics = ['R2', 'MSE', 'RMSE', 'MAE']

# Files reserved to see prediction
file_nb = 7


for list_amp in list_list_amp:
    # -------------------------------- Load data ----------------------------------------
    load_data = LoadData()
    for folder in list_amp:
        load_data.load_polymander_data(dir_name=f'logs_polymander/static/FL/{folder}')
        load_data.load_force_plates_data(dir_name=f'logs_force_plates/static/FL/{folder}')
    print(f'{len(load_data.list_polymander)} files in list_polymander')
    print(f'{len(load_data.list_force_plates)} files in list_force_plate')
    # -------------------------------- Feature engineering ----------------------------
    check = False
    count = 0
    for (i, j) in zip(load_data.list_polymander, load_data.list_force_plates):
        # signal processing
        t_s_final, fbck_position_final, fbck_current_final, Fxyz_final = prepare_data_for_training(i, j)

        # Store data in X (independent values) and y (dependent value)
        if check is False:
            X = fbck_current_final[:, 8:10]
            y = Fxyz_final[:, 2]
            check = True
        elif check is True and count != file_nb:
            X = np.concatenate((X, fbck_current_final[:, 8:10]))
            y = np.concatenate((y, Fxyz_final[:, 2]))
        elif check is True and count == file_nb:
            X_test2 = fbck_current_final[:, 8:10]
            y_test2 = Fxyz_final[:, 2]
            t_s_pred = t_s_final
            print(f'file nb {file_nb} is removed from the train/test dataset')
        count += 1

    # ------------------------ Supervised learning ----------------------------------
    # Train and test ratio 0.75
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

    # In which folder store the plots
    if list_amp == list_amp_0:
        save_folder = list_amp[0][0:7]
    elif list_amp == list_amp_1:
        save_folder = list_amp[0][0:8]
    elif list_amp == list_amp_2:
        save_folder = list_amp[0][0:7]

    # Linear regression with hip motor (motor 8)
    x_train = X_train[:, 0]
    x_test = X_test[:, 0]
    x_test2 = X_test2[:, 0]

    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)
    y_pred = lr.predict(x_test.reshape(-1, 1))
    y_pred2 = lr.predict(x_test2.reshape(-1, 1))

    # Metrics
    r2 = lr.score(x_train.reshape(-1, 1), y_train)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'lr metrics for hip current (current 8): [R^2, MSE, RMSE, MAE] = [{r2:.2f}& {mse:.2f}& {rmse:.2f}& {mae:.2f}]')

    # Plot results
    fig, ax = plt.subplots()
    ax.set_title('Linear regression of hip motor')
    ax.set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
    ax.scatter(x_train, y_train)
    ax.plot(x_train, lr.coef_*x_train + lr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title('Prediction of Fz with hip motor model')
    ax.plot(t_s_pred, y_test2, label='true value')
    ax.plot(t_s_pred, y_pred2, label='pred')
    ax.set(xlabel='time [s]', ylabel='Fz [N]')
    ax.legend()

    plt.savefig(f'figures/lr_results/amp/lr_8_{save_folder}.png', format='png')
    plt.savefig(f'figures/lr_results/amp/lr_8_{save_folder}.eps', format='eps')


    # Linear regression with calf motor (motor 9)
    x_train = X_train[:, 1]
    x_test = X_test[:, 1]
    x_test2 = X_test2[:, 1]

    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)
    y_pred = lr.predict(x_test.reshape(-1, 1))
    y_pred2 = lr.predict(x_test2.reshape(-1, 1))

    # Metrics
    r2 = lr.score(x_train.reshape(-1, 1), y_train)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'lr metrics for calf current (current 9): [R^2, MSE, RMSE, MAE] = [{r2:.2f}& {mse:.2f}& {rmse:.2f}& {mae:.2f}]')

    # Plot results
    fig, ax = plt.subplots()
    ax.set_title('Linear regression of calf motor')
    ax.set(xlabel='Feedback current [mA]', ylabel='Fz [N]')
    ax.scatter(x_train, y_train)
    ax.plot(x_train, lr.coef_*x_train + lr.intercept_, 'r', label='$R^2 = %.2f$' % r2)
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title('Prediction of Fz with calf motor model')
    ax.plot(t_s_pred, y_test2, label='true value')
    ax.plot(t_s_pred, y_pred2, label='pred')
    ax.set(xlabel='time [s]', ylabel='Fz [N]')
    ax.legend()

    plt.savefig(f'figures/lr_results/amp/lr_9_{save_folder}.png', format='png')
    plt.savefig(f'figures/lr_results/amp/lr_9_{save_folder}.eps', format='eps')

    # Multiple linear regression
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    y_pred = mlr.predict(X_test)
    y_pred2 = mlr.predict(X_test2)

    # Metrics
    r2 = mlr.score(X_train, y_train)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'mlr metrics: [R^2, MSE, RMSE, MAE] = [{r2:.2f}& {mse:.2f}& {rmse:.2f}& {mae:.2f}]')

    # K-fold cross validation
    folds = KFold(n_splits=5, shuffle=True, random_state=100)

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

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
    ax.plot(X_train[:, 0], X_train[:, 1], mlr.coef_[0]*X_train[:, 0] + mlr.coef_[1]*X_train[:, 1] + mlr.intercept_, 'r',
            label='$R^2 = %.2f$' % r2)
    ax.set(xlabel='8FbckCurrent [mA]', ylabel='9FbckCurrent [mA]', zlabel='Fz [N]')
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title('Prediction of Fz with hip and calf motors')
    ax.plot(t_s_pred, y_test2, label='true value')
    ax.plot(t_s_pred, y_pred2, label='pred')
    ax.set(xlabel='time [s]', ylabel='Fz [N]')
    ax.legend()

    plt.savefig(f'figures/lr_results/amp/mlr_{save_folder}.png', format='png')
    plt.savefig(f'figures/lr_results/amp/mlr_{save_folder}.eps', format='eps')

plt.show()



