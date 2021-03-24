#!/bin/bash

bsub -W 72:00 -u bestocke -J "ingest_modis" -R "rusage[mem=24000]" "Rscript --vanilla rscript_ingest_modis.R"
