import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
import math
import pandas as pd
import pickle
import torch

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

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

def generate_graph(mol):
    # Create Node Features
    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)
    x = torch.tensor(xs, dtype=torch.float).view(-1, 9)
    
    # Create Edge Features
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        
    return x, edge_index, edge_attr