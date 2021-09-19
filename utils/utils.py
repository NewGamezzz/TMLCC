import matplotlib.pyplot as plt
import numpy as np

def check_nan_inf(X):
    return np.isnan(X).sum()+np.isinf(X).sum()

def show_history(hist):
    plt.title('loss')
    plt.plot(hist.epoch, hist.history['loss'], label='train loss')
    plt.plot(hist.epoch, hist.history['val_loss'], label='val loss')
    plt.legend()

def clean_heat_adsorption_train_test(X_train, X_test, features_idx, num_catalog):
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
    
    return heat_adsorp_catalog

def clean_heat_adsorption(X, features_idx, num_catalog):
    catalog = X[::, features_idx['catalog CO2/N2']]
    heat_adsorp_list = X[::, features_idx['heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]']]

    heat_adsorp_catalog = {'data': {}, 'mean_std': {}}
    for c in range(num_catalog):
        heat_adsorp_catalog['data'][c] = heat_adsorp_list[catalog==c][np.logical_and(
                                                                        ~np.isnan(heat_adsorp_list[catalog==c]), 
                                                                        ~np.isinf(heat_adsorp_list[catalog==c]))]

        heat_adsorp_catalog['mean_std'][c] = heat_adsorp_catalog['data'][c].mean(), heat_adsorp_catalog['data'][c].std()

    for c in range(num_catalog):
        heat_adsorp_list[np.logical_and(np.isnan(heat_adsorp_list), catalog==c)] = heat_adsorp_catalog['mean_std'][c][0]
        heat_adsorp_list[np.logical_and(np.isinf(heat_adsorp_list), catalog==c)] = heat_adsorp_catalog['mean_std'][c][0]
    
    return heat_adsorp_catalog


def generate_selectivity_catalog(df):
    num_catalog = 2
    bound_range = [0, 7] # [0, 7, 10, 20, 40]
    count = []
    mask = []

    for i, bound in enumerate(bound_range):
        if i == len(bound_range)-1:
            m = np.logical_and(bound < df['CO2/N2_selectivity'], df['CO2/N2_selectivity'] <= np.inf)
            mask.append(m)
            count.append(m.sum())
        else:
            m = np.logical_and(bound < df['CO2/N2_selectivity'], df['CO2/N2_selectivity'] <= bound_range[i+1])
            mask.append(m)
            count.append(m.sum())

    catalog = np.zeros_like(df['CO2/N2_selectivity'].values)
    for i, m in enumerate(mask):
        catalog[m] = i

    df.insert(11, "catalog CO2/N2", catalog)
    return df

def target_encoder_train_test(df, X_train, X_test, y_train, y_test, features_idx, feature_encode):
    key_ = df[feature_encode].unique()
    feature_mean = {}
    for key in key_:
        feature_mean[key] = y_train[X_train[:, features_idx[feature_encode]]==key].mean()

    encode_train = np.zeros_like(y_train)
    encode_test = np.zeros_like(y_test)
    for key in key_:
        encode_train[X_train[:, features_idx[feature_encode]]==key] = feature_mean[key]
        encode_test[X_test[:, features_idx[feature_encode]]==key] = feature_mean[key]

    X_train[:, features_idx[feature_encode]] = encode_train.squeeze()
    X_test[:, features_idx[feature_encode]] = encode_test.squeeze()
    return feature_mean

def target_encoder(df, X, y, features_idx, feature_encode):
    key_ = df[feature_encode].unique()
    feature_mean = {}
    for key in key_:
        feature_mean[key] = y[X[:, features_idx[feature_encode]]==key].mean()

    encode = np.zeros_like(y)
    for key in key_:
        encode[X[:, features_idx[feature_encode]]==key] = feature_mean[key]

    X[:, features_idx[feature_encode]] = encode.squeeze()
    return feature_mean