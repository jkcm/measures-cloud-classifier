#!/bin/csh
#PBS -l nodes=1:ppn=12
#PBS -l walltime=12:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N MOD02_dl_2

conda activate py37
cd /home/disk/p/jkcm/Code/measures-cloud-classifier
which python >>& run_downloader.log
python download_and_subset_modis.py -s 20151101 -e 20160331 >>& logs/mod02_download.2.log
exit 0
