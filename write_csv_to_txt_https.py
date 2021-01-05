
from __future__ import print_function, division
import numpy as np
import os,csv,glob

def find_files_Terra(csvreader):
    Satellite   = 'Terra'         # Satellite name
    MOD06_fn,MOD03_fn,MOD02_fn,date_fn = [],[],[],[]    # Initialize input file names and date    
    MOD03_fns,MOD02_fns = [],[]

    for row in csvreader:
        if(len(row)>2 and 'MOD06_L2' in row[1] and '.hdf' in row[1]): MOD06_fn.append(row[1])
        if(len(row)>2 and 'MOD03' in row[1] and '.hdf' in row[1]): MOD03_fns.append(row[1])
        if(len(row)>2 and 'MOD021KM' in row[1] and '.hdf' in row[1]): MOD02_fns.append(row[1])

    n_files = (len(MOD06_fn))     # Total number of MOD06_L2 files
    if(n_files > 0):
      for nf in range(n_files):
          fname = os.path.basename(MOD06_fn[nf])      
          date  = fname.split('.')[1]+'.'+fname.split('.')[2]+'.'+fname.split('.')[3]
          date_fn.append(date)
          if([s for s in MOD03_fns if date in s]):
             MOD03_fn.append([s for s in MOD03_fns if date in s][0])
          else: MOD03_fn.append('')
          if([s for s in MOD02_fns if date in s]):
             MOD02_fn.append([s for s in MOD02_fns if date in s][0])
          else: MOD02_fn.append('')
    return MOD02_fn,MOD03_fn,MOD06_fn,date_fn,n_files

def find_files_Aqua(csvreader):
    Satellite   = 'Aqua'          # Satellite name
    MYD06_fn,MYD03_fn,MYD02_fn,date_fn = [],[],[],[]    # Initialize input file names and date
    MYD03_fns,MYD02_fns = [],[]

    for row in csvreader:
        if(len(row)>2 and 'MYD06_L2' in row[1] and '.hdf' in row[1]): MYD06_fn.append(row[1])
        if(len(row)>2 and 'MYD03' in row[1] and '.hdf' in row[1]): MYD03_fns.append(row[1])
        if(len(row)>2 and 'MYD021KM' in row[1] and '.hdf' in row[1]): MYD02_fns.append(row[1])

    n_files = (len(MYD06_fn))     # Total number of MYD06_L2 files
    if(n_files > 0):
      for nf in range(n_files):
          fname = os.path.basename(MYD06_fn[nf]) 
          date  = fname.split('.')[1]+'.'+fname.split('.')[2]+'.'+fname.split('.')[3]
          date_fn.append(date)
          if([s for s in MYD03_fns if date in s]):
             MYD03_fn.append([s for s in MYD03_fns if date in s][0])
          else: MYD03_fn.append('')
          if([s for s in MYD02_fns if date in s]):
             MYD02_fn.append([s for s in MYD02_fns if date in s][0])
          else: MYD02_fn.append('')
    return MYD02_fn,MYD03_fn,MYD06_fn,date_fn,n_files

def convert_csv_to_wget_file(csv_file):
    Data_path     = 'https://ladsweb.modaps.eosdis.nasa.gov'                        # http path 
    newfile = csv_file.replace('.csv','_https.txt')
    with open(csv_file) as csvfile:
        print('Input csv file: ',os.path.basename(csv_file))
        csvreader    = csv.reader(csvfile,delimiter=',')
        csv_headings = next(csvreader)
        csv_headings = next(csvreader)
        n_files = 0    # Number of M?D06 files in each csv file
        MD02_fn,MD03_fn,MD06_fn,date_fn = [],[],[],[]    # Initialize key variables of input data
        print(csv_headings)
        if(any('MOD' in string for string in csv_headings)):
            MD02_fn,MD03_fn,MD06_fn,date_fn,n_files = find_files_Terra(csvreader)
        if(any('MYD' in string for string in csv_headings)):
            print('MYD files')
            MD02_fn,MD03_fn,MD06_fn,date_fn,n_files = find_files_Aqua(csvreader)
        if(n_files > 0):
            with open(newfile, 'w') as list_files:
                print('writing to {}'.format(newfile))
                #fnames_list = CSV_path + 'txt/' + os.path.basename(csv_file).replace('.csv','_https.txt')
                #open(fnames_list,'a').close()
                for n_file in range(n_files):             
                    #list_files  = open(fnames_list,'a')
                    if MD06_fn[n_file] is not '':
                        list_files.writelines(Data_path+MD06_fn[n_file]+'\n')
                    if MD03_fn[n_file] is not '':
                        list_files.writelines(Data_path+MD03_fn[n_file]+'\n')
                    if MD02_fn[n_file] is not '':
                        list_files.writelines(Data_path+MD02_fn[n_file]+'\n')
        else:
            print('no files found!')






# Beginning of the main program
if __name__ == "__main__":

  # Define paths for input data 
#   CSV_path      = '/home/disk/eos4/jkcm/Data/MEASURES/MODIS_downloads/sample2/'   # Path for CSV files   
  CSV_path = r'/home/disk/eos9/jkcm/Data/modis/SEP/'
  CSV_Data_path = '/archive/allData/61/'                                          # Data path in the CSV files 
  Data_path     = 'https://ladsweb.modaps.eosdis.nasa.gov'                        # http path 
       
  # CSV files that include the data path and file names
  csv_files = sorted(glob.glob(CSV_path+'*.csv'))
  print(csv_files)
  if(csv_files):
    for csv_file in csv_files:
    # Read input data filenames from CSV files
      with open(csv_file) as csvfile:
         print('Input csv file: ',os.path.basename(csv_file))
         csvreader    = csv.reader(csvfile,delimiter=',')
         csv_headings = next(csvreader)
         csv_headings = next(csvreader)
         n_files = 0    # Number of M?D06 files in each csv file
         MD02_fn,MD03_fn,MD06_fn,date_fn = [],[],[],[]    # Initialize key variables of input data

         if(any('MOD' in string for string in csv_headings)):
            MD02_fn,MD03_fn,MD06_fn,date_fn,n_files = find_files_Terra(csvreader)

         if(any('MYD' in string for string in csv_headings)):
            MD02_fn,MD03_fn,MD06_fn,date_fn,n_files = find_files_Aqua(csvreader)

         if(n_files > 0):
           fnames_list = CSV_path + 'txt/' + os.path.basename(csv_file).replace('.csv','_https.txt')
           open(fnames_list,'a').close()
           for n_file in range(n_files):             
               list_files  = open(fnames_list,'a')
               list_files.writelines(Data_path+MD06_fn[n_file]+'\n')
               list_files.writelines(Data_path+MD03_fn[n_file]+'\n')
               list_files.writelines(Data_path+MD02_fn[n_file]+'\n')
               list_files.close()
         else:
            print('no files found')

