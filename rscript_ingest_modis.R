#!/usr/bin/env Rscript

library(tidyverse)
library(ingestr)

ncores <- parallel::detectCores()

df_sites <- read_csv("./data/df_sites.csv")

settings_modis <- get_settings_modis(
  bundle            = "modis_fpar",
  data_path         = "~/data/modis_subsets/",
  method_interpol   = "loess",
  keep              = TRUE,
  overwrite_raw     = FALSE,
  overwrite_interpol= TRUE
)

df_modis_fpar <- ingest(
  df_sites,
  source = "modis",
  settings = settings_modis,
  parallel = FALSE,
  ncores = ncores
)
