# mlflx: Machine learning for FLUXNET data.

## Motivation

Ecosystem-atmosphere exchange fluxes of water vapour and CO2 are continuously measured at several hundred of sites, distributed across the globe. The oldest running sites have been recording data since over twenty years. Thanks to the international FLUXNET initiative, these time series data are made openly accessible from over hundred sites and provided in a standardized format and complemented with measurements of several meteorological variables, plus soil temperature and moisture, measured in parallel. These data provide an opportunity for understanding ecosystem fluxes and how they are affected by environmental covariates. The challenge is to build models that are sufficiently generalisable in space. That is, temporally resolved relationships learned from one subset of sites should be used effectively to predict time series, given environmental covariates, at new sites (spatial upscaling). This is a challenge as previous research has shown that relatively powerful site-specific models can be trained, but predictions to new sites have been found wanting. This may be due to site-specific characteristics (e.g. vegetation type) that have thus far not been satisfactorily encoded in models. In other words, factors that would typically be regarded as random factors in mixed effects modelling, continue to undermine effective learning in machine learning models. 

## Data

Data is provided here at daily resolution from a selection of sites ($N=71$, 265,177 data points, see `prepare_data.Rmd`), and paired with satellite data of the fraction of absorbed photosynthetically active radiation (fAPAR, product MODIS FPAR). This provides crucial complimentary information about vegetation structure and seasonally varying green foliage cover, responsible for photosynthesis and transpiration. The target variable is ecosystem photosynthesis, referred to as gross primary production (variable `GPP_NT_VUT_REF`). Available covariates are briefly described below. For more information, see [FLUXNET 2015 website](http://fluxnet.fluxdata.org/data/fluxnet2015-dataset/), and [Pastorello et al., 2020](https://www.nature.com/articles/s41597-020-0534-3), and the document `variable_codes_FULLSET_20160711.pdf` provided in this repository.

The data is provided through this repository (`data/ddf_combined_mlflx.csv`).

![Site selection](./fig/map_sites_mlflx.png)

### Available variables

- `sitename`: FLUXNET standard site name
- `date`
- `fpar_loess`: fraction of absorbed photosynthetically active radiation, interpolated to daily values using LOESS.
- `fpar_linear`: fraction of absorbed photosynthetically active radiation, linearly interpolated to daily values.
- `TA_F`: Air temperature. The meaning of suffix `_F` is described in [Pastorello et al., 2020](https://www.nature.com/articles/s41597-020-0534-3).
- `SW_IN_F`: Shortwave incoming radiation
- `LW_IN_F`: Longwave incoming radiation
- `VPD_F`: Vapour pressure deficit (relates to the humidity of the air)
- `PA_F`: Atmospheric pressure
- `P_F`: Precipitation
- `WS_F`: Wind speed
- `USTAR`: A measure for atmospheric stability
- `CO2_F_MDS`: CO2 concentration
- `GPP_NT_VUT_REF`: Gross primary production - **the target variable**
- `NEE_VUT_REF_QC`: Quality control information for `GPP_NT_VUT_REF`
- `SWC_F_MDS_*`: Soil water content. The number provides information about the soil layer, counting from the top
- `lon`, `lat`: Longitude and latitude (degrees)

## The challenge

Formulate a model `GPP_NT_VUT_REF ~ `, selecting predictors as suitable. Conceive the model training such that validation and test sets are split along sites (data from a given site must not be both in the test and training sets). Demonstrate the spatial generalisability of the trained model.

