#!/bin/csh
#PBS -l nodes=1:ppn=8
#PBS -l walltime=8:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N MOD06_dl

conda activate py37
cd /home/disk/p/jkcm/Code/measures-cloud-classifier
python download_and_subset_modis.py -p MYD06_L2 -e 20150831 >>& logs/mod06_download.log
exit 0
