#!/bin/bash

module load eth_proxy

# njobs=69
# for ((n=1;n<=${njobs};n++)); do
#     echo "Submitting chunk number $n ..."
#     bsub -W 72:00 -u bestocke -J "ingest_modis_reflectance $n" -R "rusage[mem=5000]" "Rscript --vanilla ingest_modis_reflectance.R $n"
# done

bsub -W 72:00 -u bestocke -J "ingest_modis_reflectance $n" -R "rusage[mem=5000]" "Rscript --vanilla ingest_modis_reflectance.R 5"
