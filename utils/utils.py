import matplotlib.pyplot as plt
import numpy as np

def check_nan_inf(X):
    return np.isnan(X).sum()+np.isinf(X).sum()

def show_history(hist):
    plt.title('loss')
    plt.plot(hist.epoch, hist.history['loss'], label='train loss')
    plt.plot(hist.epoch, hist.history['val_loss'], label='val loss')
    plt.legend()

def clean_heat_adsorption(X_train, X_test, features_idx, num_catalog):
    # Clean Train
    catalog = X_train[::, features_idx['catalog CO2/N2']]
    heat_adsorp_list = X_train[::, features_idx['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]']]

    heat_adsorp_catalog = {'data': {}, 'mean_std': {}}
    for c in range(num_catalog):
        heat_adsorp_catalog['data'][c] = heat_adsorp_list[catalog==c][np.logical_and(
                                                                        ~np.isnan(heat_adsorp_list[catalog==c]), 
                                                                        ~np.isinf(heat_adsorp_list[catalog==c]))]

        heat_adsorp_catalog['mean_std'][c] = heat_adsorp_catalog['data'][c].mean(), heat_adsorp_catalog['data'][c].std()

    for c in range(num_catalog):
        heat_adsorp_list[np.logical_and(np.isnan(heat_adsorp_list), catalog==c)] = heat_adsorp_catalog['mean_std'][c][0]
        heat_adsorp_list[np.logical_and(np.isinf(heat_adsorp_list), catalog==c)] = heat_adsorp_catalog['mean_std'][c][0]

    # Clean Test
    catalog = X_test[::, features_idx['catalog CO2/N2']]
    heat_adsorp_list = X_test[::, features_idx['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]']]

    for c in range(num_catalog):
        heat_adsorp_list[np.logical_and(np.isnan(heat_adsorp_list), catalog==c)] = heat_adsorp_catalog['mean_std'][c][0]
        heat_adsorp_list[np.logical_and(np.isinf(heat_adsorp_list), catalog==c)] = heat_adsorp_catalog['mean_std'][c][0]