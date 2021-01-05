#!/bin/csh
#PBS -l nodes=1:ppn=8
#PBS -l walltime=0:05:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N MODIS_dm

cd /home/disk/p/jkcm/Code/measures-cloud-classifier
/home/disk/p/jkcm/anaconda3/bin/python dummy.py >& dummy.log
exit 0
