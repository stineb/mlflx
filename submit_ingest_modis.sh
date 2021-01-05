#!/bin/bash

bsub -W 72:00 -u bestocke -J "ingest_modis" -R "span[ptile=36]" "Rscript --vanilla rscript_ingest_modis.R"
