import numpy as np

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result


def prepare_df(data, target = "GRA"):
    # Site Data
    sensor_data = data.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    sources = data[data.classid != target].classid.unique()

    # Batch by  source and site
    df_sensor_s = []
    df_gpp_s = []
    df_domain_s = []
    for k in sources:
        sites = data[data.classid == k].index.unique()
        df_sensor = [normalize(sensor_data[sensor_data.index == site]) for site in sites if sensor_data[sensor_data.index == site].size != 0 ]
        df_gpp = [data[data.index == site]['GPP_NT_VUT_REF'] for site in sites if data[data.index == site].size != 0]   
        df_gpp = [(df_gpp[i]-df_gpp[i].mean())/df_gpp[i].std() for i in range(len(df_gpp))]
        df_sensor_s.append(df_sensor)
        df_gpp_s.append(df_gpp)
        df_domain_s.append(0)
        
    sites = data[data.classid == target].index.unique()
    df_sensor_t = [normalize(sensor_data[sensor_data.index == site]) for site in sites if sensor_data[sensor_data.index == site].size != 0 ]
    df_gpp_t = [data[data.index == site]['GPP_NT_VUT_REF'] for site in sites if data[data.index == site].size != 0]   
    df_gpp_t = [(df_gpp_t[i]-df_gpp_t[i].mean())/df_gpp_t[i].std() for i in range(len(df_gpp_t))]
    df_domain_t = [1 for site in sites]


    # Mask imputed data
    raw = pd.read_csv('../data/df_20210510.csv', index_col=0)
    raw = raw[raw.classid == target]
    raw = raw[raw.index != 'CN-Cng']['GPP_NT_VUT_REF']
    
    sites = raw.index.unique()
    masks = []
    for s in sites:
        mask = raw[raw.index == s].isna().values
        masks.append(list(map(operator.not_, mask)))

    return df_sensor_s, df_sensor_t, df_gpp_s, df_gpp_t, df_domain_s, df_domain_t, masks