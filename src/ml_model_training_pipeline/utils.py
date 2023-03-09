import numpy as np
import pandas as pd
import tensorflow as tf


def extract_data(csv_url, column_name):
    raw_dataset = pd.read_csv(csv_url, names=column_name,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    return raw_dataset


def get_feature_normalizer(features: np.ndarray) -> tf.keras.layers.Normalization:
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(features)

    return normalizer
