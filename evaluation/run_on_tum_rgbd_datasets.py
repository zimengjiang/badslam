#!/usr/bin/env python

import errno
import os
import subprocess
import sys
import timeit


### ADJUST THIS TO YOUR ENVIRONMENT ###
tum_rgbd_datasets_path = '/local/home/zjiang/data/tum_rgbd/'  # the folder containing the training and test folders
slam_program_path = '/local/home/zjiang/fbadslam/build/applications/badslam/badslam'

### ------------------------------- ###


program_with_arguments = []
return_code = 0


# Converts a string to bytes (for writing the string into a file). Provided for
# compatibility with Python 2 and 3.
def StrToBytes(text):
    if sys.version_info[0] == 2:
        return text
    else:
        return bytes(text, 'UTF-8')

# Creates the given directory (hierarchy), which may already exist. Provided for
# compatibility with Python 2 and 3.
def MakeDirsExistOk(directory_path):
    try:
        os.makedirs(directory_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def RunMethod():
    proc = subprocess.Popen(program_with_arguments)
    return_code = proc.wait()

if __name__ == '__main__':
    # Check that the paths have been set
    if len(slam_program_path) == 0 or len(tum_rgbd_datasets_path) == 0:
        print('Before using this script, please set slam_program_path in this script to the path of the' +
              ' SLAM executable that you want to run, and set tum_rgbd_datasets_path' +
              ' in this script to the path of the tum_rgbd datasets on your system.')
        sys.exit(1)
    
    # Print the paths such that the user is aware of them
    print('Datasets path: ' + tum_rgbd_datasets_path)
    print('SLAM program: ' + slam_program_path)
    
    # Parse arguments
    if len(sys.argv) < 3:
        print('Usage:')
        print('python run_on_tum_rgbd_datasets.py training/test result_name [additional_options_to_slam_program]')
        print('')
        sys.exit(1)
    
    datasets_folder = ''
    if sys.argv[1] == 'training':
        datasets_folder = os.path.join(tum_rgbd_datasets_path, 'training')
    elif sys.argv[1] == 'test':
        datasets_folder = os.path.join(tum_rgbd_datasets_path, 'test')
    else:
        print('Either \'training\' or \'test\' must be given as first argument to the script.')
        sys.exit(1)
    
    result_name = sys.argv[2]
    additional_arguments = sys.argv[3:]
    
    # Check that the benchmark is there
    if not os.path.isdir(datasets_folder):
        print('The folder does not exist: ' + datasets_folder)
        sys.exit(1)
    
    # Loop over all datasets
    for item in os.listdir(datasets_folder):
        dataset_path = os.path.join(datasets_folder, item)
        if not os.path.isdir(dataset_path):
            continue
        
        # Check for the presence of the mono dataset files
        if (not os.path.isdir(os.path.join(dataset_path, 'rgb')) or
            not os.path.isfile(os.path.join(dataset_path, 'calibration.txt')) or
            not os.path.isfile(os.path.join(dataset_path, 'rgb.txt'))):
            print('Folder does not seem to contain a dataset, skipping: ' + dataset_path)
            continue
        
        print('Dataset: ' + dataset_path)
        
        # Create the results folder
        results_folder = os.path.join(dataset_path, 'results')
        MakeDirsExistOk(results_folder)
        
        result_trajectory_path = os.path.join(results_folder, result_name + '.txt')
        result_runtime_path = os.path.join(results_folder, result_name + '_runtime.txt')
        
        # Run the SLAM program on this dataset
        program_with_arguments = [slam_program_path]
        program_with_arguments.append(dataset_path)
        program_with_arguments.append('--export_poses')
        program_with_arguments.append(result_trajectory_path)
        program_with_arguments.extend(additional_arguments)
        
        print('Running: ' + ' '.join(program_with_arguments))
        # RunMethod() will access program_with_arguments and return_code since they are globals
        runtime = timeit.timeit(stmt = "RunMethod()", setup = "from __main__ import RunMethod", number = 1)
        if return_code != 0:
            print('Algorithm call failed (return code: ' + str(return_code) + ')')
        
        # In case the program did not write a result file, write an empty one
        if not os.path.isfile(result_trajectory_path):
            result_file = open(result_trajectory_path, 'wb')
            result_file.close()
        
        # Write runtime file
        with open(result_runtime_path, 'wb') as outfile:
            outfile.write(StrToBytes(str(runtime)))
