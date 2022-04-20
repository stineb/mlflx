#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)

library(tidyverse)
library(ingestr)  # https://github.com/stineb/ingestr

load("data/ddf_fluxnet_mlflx_wscal_20210505.RData")
df_sites <- read_csv("./data/df_sites_mlflx.csv")

## create df_sites based on ddf to make sure we get the right dates matching the fluxnet data we now have nicely processed
df_sites_ddf <- ddf %>% 
  mutate(year = lubridate::year(TIMESTAMP)) %>% 
  group_by(sitename) %>% 
  summarise(year_start = min(year), year_end = max(year)) %>% 
  left_join(df_sites %>% dplyr::select(sitename, lon, lat),
            by = "sitename")

settings_modis <- get_settings_modis(
  bundle            = "modis_refl_terra",
  data_path         = "~/data/modis_subsets/",
  method_interpol   = "loess",
  keep              = TRUE,
  overwrite_raw     = FALSE,
  overwrite_interpol= TRUE,
  n_focal           = 0
)

df_modis_fpar <- ingest(
  df_sites_ddf %>% 
    mutate(year_start = 2000, year_end = 2020) %>% 
    slice(as.numeric(args[1])),
  source = "modis",
  settings = settings_modis, 
  parallel = FALSE
  # ncores = 2
)