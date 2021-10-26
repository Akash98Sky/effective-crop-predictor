import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs

EACH_SET = 100

def create_feature(num, center, std):
    return make_blobs(num, 1, 1, center_box=center, cluster_std=std)


def crop_data(ph, n, p, k, temp, rain, std):
    if len(std) != 6:
        print('std must be of len 6')

    pH = create_feature(EACH_SET, ph, std[0])[0]
    N = create_feature(EACH_SET, n, std[2])[0]
    P = create_feature(EACH_SET, p, std[3])[0]
    K = create_feature(EACH_SET, k, std[4])[0]
    Temp = create_feature(EACH_SET, temp, std[1])[0]
    Rain = create_feature(EACH_SET, rain, std[5])[0]

    return np.concatenate((pH, N, P, K, Temp, Rain), axis=1)


def build_dataset(crop_features: list):
    dset = None
    for i in range(len(crop_features)):
        tmp = np.zeros([len(crop_features[i]), 1])
        tmp.fill(i)

        crop_features[i] = np.concatenate([crop_features[i], tmp], axis=1)

        if dset is None:
            dset = crop_features[i]
        else:
            dset = np.concatenate([dset, crop_features[i]])

    return dset


features = ['pH', 'N', 'P', 'K', 'Temp', 'Rain', 'Crop']


cotton = crop_data(ph=[7.6, 8.0], n=[110, 110], p=[45, 45], k=[50, 50], temp=[
                   29, 29],  rain=[750, 1200], std=[0.2, 8.0, 4.0, 5.0, 0.6, 50.0])


sugar_cane = crop_data(ph=[6.8, 7.1], n=[175, 175], p=[100, 100], k=[100, 100], temp=[
    32, 32],  rain=[750, 1200], std=[0.3, 8.0, 4.0, 5.0, 0.6, 50.0])

jowar = crop_data(ph=[6.3, 8.0], n=[85, 85], p=[35, 35], k=[45, 45], temp=[
    26, 27],  rain=[850, 1000], std=[0.2, 8.0, 4.0, 5.0, 0.6, 50.0])

bajra = crop_data(ph=[7.6, 8.2], n=[50, 50], p=[25, 25], k=[20, 20], temp=[
    30, 31],  rain=[350, 750], std=[0.3, 8.0, 4.0, 5.0, 0.6, 50.0])

soyabeans = crop_data(ph=[6.9, 7.4], n=[25, 25], p=[70, 70], k=[20, 20], temp=[
    30, 30],  rain=[750, 1000], std=[0.3, 8.0, 4.0, 5.0, 0.6, 50.0])

corn = crop_data(ph=[7.5, 8.0], n=[90, 90], p=[25, 25], k=[10, 10], temp=[
    35, 35],  rain=[400, 600], std=[0.3, 8.0, 4.0, 5.0, 0.6, 50.0])

rice = crop_data(ph=[6.6, 8.2], n=[100, 100], p=[50, 50], k=[50, 50], temp=[
    19, 20],  rain=[50, 200], std=[0.3, 8.0, 4.0, 5.0, 0.6, 50.0])

wheat = crop_data(ph=[5.7, 8.2], n=[110, 110], p=[50, 50], k=[50, 50], temp=[
    22, 23],  rain=[800, 1400], std=[0.3, 8.0, 4.0, 5.0, 0.6, 50.0])

ground_nut = crop_data(ph=[6.3, 7.6], n=[30, 30], p=[50, 50], k=[50, 50], temp=[
    26, 27],  rain=[600, 1200], std=[0.4, 8.0, 4.0, 5.0, 0.6, 50.0])


df = pd.DataFrame(build_dataset([cotton, sugar_cane, jowar, bajra,
                            soyabeans, corn, rice, wheat, ground_nut]), columns=features)

df.to_csv('crop_new.csv')
