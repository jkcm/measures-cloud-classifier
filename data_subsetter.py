# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:42:46 2018

@author: jkcm
"""

import pandas as pd
import os
import random
from shutil import copy

#parameters
root_directory = r'/home/disk/eos4/jkcm/Data/MEASURES/sample_root_dir'
desired_number_of_scenes_per_region_and_season = 1000
minimum_valid_scenes_per_granule = 20

def make_new_manifest(manifest, unique_granules):
    new_manifest = pd.DataFrame(columns=manifest.columns)
    print('loading {} scenes into new manifest...\n'.format(desired_number_of_scenes_per_region_and_season))
    while len(new_manifest) < desired_number_of_scenes_per_region_and_season:
        new_granule, count = random.choice(list(unique_granules.items()))
        if count < minimum_valid_scenes_per_granule: # don't include granules with only a few valid scenes. NOTE: This might introduce a bias?  
            del unique_granules[new_granule]
            continue
        print(len(new_manifest), end="\r")
        matching_rows = manifest[manifest['granule'] == new_granule]
        new_manifest = new_manifest.append(matching_rows, ignore_index=True)
        del unique_granules[new_granule]
    return(new_manifest)

def subsample_from_directory(root_directory):

    all_regions_and_seasons_dirs = [i for i in os.listdir(root_directory) if i[:4]=='2012'] # hardcoding in a pattern match here
    new_root_directory = os.path.join(root_directory, 'new_subset_{}_members'.format(desired_number_of_scenes_per_region_and_season))
    if os.path.exists(new_root_directory):
        new_root_directory += '_'+(str(random.getrandbits(32)))
        print('new root directory already exists, making a random one at {}'.format(os.path.basename(new_root_directory)))
    os.makedirs(new_root_directory)

    for folder in all_regions_and_seasons_dirs:
        print('working on ' + folder)
        os.makedirs(os.path.join(new_root_directory, folder))
        manifest_file = os.path.join(root_directory, folder, folder+'_manifest.csv')
        manifest = pd.read_csv(manifest_file)
        manifest['granule'] = [i[:22] for i in manifest['name']]
        granules = [i[:22] for i in manifest['name']]
        unique_granules = {i: 0 for i in set(granules)}
        for g in granules:
            unique_granules[g] += 1
        #creating new manifest from subselection
        new_manifest = make_new_manifest(manifest, unique_granules)
        print('writing manifest...')

        new_manifest.to_csv(os.path.join(new_root_directory, folder, folder+'_manifest.csv'), index=False)

        #copying over files
        files_to_copy = np.concatenate((new_manifest['context_img'].values, new_manifest['refl_img'].values))
        print("copying over {} files...".format(len(files_to_copy)))
        for f in files_to_copy:
            copy(os.path.join(root_directory, folder, f), os.path.join(new_root_directory, folder, f))
        print("done")

if __name__ == "__main__":
    subsample_from_directory(root_directory)
