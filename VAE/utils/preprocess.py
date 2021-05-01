from torch.utils.data import Dataset
import pandas as pd


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result


def prepare_df(data, meta_columns=['plant_functional_type','classid','koeppen_code','igbp_land_use']):
    # Site Data
    meta_data = pd.get_dummies(data[meta_columns])
    sensor_data = data.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    
    # Batch by site
    sites = data.index.unique()
    df_sensor = [normalize(sensor_data[sensor_data.index == site]) for site in sites]
    df_meta = [meta_data[meta_data.index == site] for site in sites]
    df_gpp = [data[data.index == site]['GPP_NT_VUT_REF'] for site in sites]   
    df_gpp = [(df_gpp[i]-df_gpp[i].mean())/df_gpp[i].std() for i in range(len(df_gpp))]
    return df_sensor, df_meta, df_gpp

