#!/usr/bin/env python

import errno
import os
import shutil
import sys
import tempfile


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:')
        print('python zip_eth3d_results.py path_to_benchmark training/test/all result_name')
        print('')
        sys.exit(1)
    
    # Parse arguments
    path_to_benchmark = sys.argv[1]
    
    if sys.argv[2] != 'training' and sys.argv[2] != 'test' and sys.argv[2] != 'all':
        print('Either \'training\', \'test\', or \'all\' must be given as second argument to the script.')
        sys.exit(1)
    
    result_name = sys.argv[3]
    
    # Zip the results
    temp_dir_path = tempfile.mkdtemp()
    slam_results_path = os.path.join(temp_dir_path, 'slam')
    os.makedirs(slam_results_path)
    
    folder_list = []
    if sys.argv[2] != 'training':
        folder_list.append('test')
    if sys.argv[2] != 'test':
        folder_list.append('training')
    
    for folder in folder_list:
        folder_path = os.path.join(path_to_benchmark, folder)
        for dataset_name in os.listdir(folder_path):
            dataset_path = os.path.join(folder_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            
            results_folder = os.path.join(dataset_path, 'results')
            if not os.path.isdir(results_folder):
                continue
            
            result_trajectory_path = os.path.join(results_folder, result_name + '.txt')
            result_runtime_path = os.path.join(results_folder, result_name + '_runtime.txt')
            
            if not os.path.isfile(result_trajectory_path) or not os.path.isfile(result_runtime_path):
                print('Warning: No complete result found for dataset: ' + dataset_name + ' (path: ' + dataset_path + ')')
                continue
            
            shutil.copy2(result_trajectory_path,
                         os.path.join(slam_results_path, dataset_name + '.txt'))
            shutil.copy2(result_runtime_path,
                         os.path.join(slam_results_path, dataset_name + '_runtime.txt'))
    
    archive_path = shutil.make_archive(os.path.join(path_to_benchmark, result_name), 'zip', temp_dir_path)
    print('Created archive: ' + archive_path)
    
    # Clean up
    shutil.rmtree(temp_dir_path)
