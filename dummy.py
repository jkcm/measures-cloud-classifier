from dask_jobqueue import PBSCluster
import time
from distributed import Client


#PBS -l nodes=1:ppn=8
#PBS -l walltime=0:05:00
#PBS -m ae
#PBS -M jkcm@uw.edu
# PBS -N MODIS_dm


"""
Olympus: (20 nodes)
per node: 4GB, 8 processors
AMD Opteron 2350 (2x quad core)
20Gbit/sec Infiniband 

Challenger E5520: (10 nodes)
per node: 12GB, 8 processors
Xeon E5520

Challenger E5645: (16 nodes):
per node: 12GB, 12 processors (hyperthreated to 24)
Xeon E5645
40Gbit/sec  Infiniband
"""


def silly_function(i):
    import socket
    host = socket.gethostname()
    print(host)
    time.sleep(10+i/10)
    print(i)
    return hostname
    
if __name__ == "__main__":

    
    
    
    oly_cluster = PBSCluster(
        shebang='#!/bin/tcsh',
        name='MODIS_dl',
        cores=8,
        memory='4GB',
        processes=8, #doesn't work with scale(jobs) for some reason
        local_directory='/home/disk/eos9/jkcm/temp',
        resource_spec='nodes=1:ppn=8',
        walltime='00:05:00',
        job_extra= ['-m ae', '-M jkcm@uw.edu'],
        interface='ib0'
        )
    oly_cluster.scale(4)
    print(oly_cluster.job_script())

    client = Client(oly_cluster)
    L = client.map(silly_function, range(20))
    client.gather(L)
    print(L)
#     g = 
#     print(g)