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

# -------------------------------- Load data ----------------------------------
FL_limb = LoadData()
for folder in list_folder:
    FL_limb.load_polymander_data(dir_name=f'logs_polymander/static/FL/{folder}')
    FL_limb.load_force_plates_data(dir_name=f'logs_force_plates/static/FL/{folder}')
print(f'{len(FL_limb.list_polymander)} files in list_polymander')
print(f'{len(FL_limb.list_force_plates)} files in list_force_plate')

# -------------------------------- Feature engineering ----------------------------
check = False
for (i, j) in zip(FL_limb.list_polymander, FL_limb.list_force_plates):
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

# ------------------------- Supervised learning ----------------------------------------
# Split the data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

# Create and train model - multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Save model
pickle.dump(mlr, open('models/mlr.pkl', 'wb'))

# K-fold cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=0)

r2_scores = cross_val_score(mlr, X_train, y_train, scoring='r2', cv=kf)
mse_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
rmse_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kf)
mae_scores = cross_val_score(mlr, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)

mse_scores = abs(mse_scores)
rmse_scores = abs(rmse_scores)
mae_scores = abs(mae_scores)

list_metrics = [r2_scores, mse_scores, rmse_scores, mae_scores]

print('\n------------------------ K-fold cross validation ----------------------')
print("R2: mean score of %0.4f with a standard deviation of %0.4f" % (r2_scores.mean(), r2_scores.std()))
print("MSE: mean score of %0.4f with a standard deviation of %0.4f" % (mse_scores.mean(), mse_scores.std()))
print("RMSE: mean score of %0.4f with a standard deviation of %0.4f" % (rmse_scores.mean(), rmse_scores.std()))
print("MAE: mean score of %0.4f with a standard deviation of %0.4f\n" % (mae_scores.mean(), mae_scores.std()))

# Plot metrics
fix, ax = plt.subplots()
ax.boxplot(list_metrics)
ax.set_title('Multiple linear regression model')
ax.set(xlabel='Metrics')
ax.set_xticks([1, 2, 3, 4], ['R2', 'MSE', 'RMSE', 'MAE'])

plt.savefig('figures/lr_results/one_model_metrics.png', format='png')

plt.show()
