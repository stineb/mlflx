import pandas as pd

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != 'GPP_NT_VUT_REF':
          result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result

def batch_by_site(df):
    sites = df.index.unique()
    sites_df = [df[df.index.isin([site])] for site in sites]
    for i in range(len(sites_df)):
        sites_df[i]['date'] = pd.to_datetime(sites_df[i]['date'], format="%Y-%m-%d")
        sites_df[i] = sites_df[i].set_index("date")
        sites_df[i] = sites_df[i].drop(columns=["lon","lat","plant_functional_type","igbp_land_use","koeppen_code","classid","c4"])
        sites_df[i] = sites_df[i].reset_index(drop=True)
    return sites_df

