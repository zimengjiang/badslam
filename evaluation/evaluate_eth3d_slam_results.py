#!/usr/bin/env python

import datetime
import errno
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import sys


### ADJUST THIS TO YOUR ENVIRONMENT ###
eth3d_datasets_path = '/local/home/zjiang/data/eth3d/'  # the folder containing the training and test folders
evaluation_program_path = '/local/home/zjiang/slam-evaluation/build/ETH3DSLAMEvaluation'
### ------------------------------- ###


# Tuples: (metric_tag, metric_filename, metric_latex)
metrics = [('SE3_ATE_RMSE[cm]:', 'se3_ate_rmse', 'SE(3) ATE RMSE [cm]'),
           ('SE3_REL_TRANSLATION_0.5M[%]:', 'se3_rel_translation_0_5', 'SE(3) rel. translation (0.5m) [\\%]'),
           ('SE3_REL_ROTATION_0.5M[deg/m]:', 'se3_rel_rotation_0_5', 'SE(3) rel. rotation (0.5m) [$\\frac{deg}{m}$]'),
           ('SE3_REL_TRANSLATION_1M[%]:', 'se3_rel_translation_1_0', 'SE(3) rel. translation (1m) [\\%]'),
           ('SE3_REL_ROTATION_1M[deg/m]:', 'se3_rel_rotation_1_0', 'SE(3) rel. rotation (1m) [$\\frac{deg}{m}$]'),
           ('SE3_REL_TRANSLATION_1.5M[%]:', 'se3_rel_translation_1_5', 'SE(3) rel. translation (1.5m) [\\%]'),
           ('SE3_REL_ROTATION_1.5M[deg/m]:', 'se3_rel_rotation_1_5', 'SE(3) rel. rotation (1.5m) [$\\frac{deg}{m}$]'),
           ('SE3_REL_TRANSLATION_2M[%]:', 'se3_rel_translation_2_0', 'SE(3) rel. translation (2m) [\\%]'),
           ('SE3_REL_ROTATION_2M[deg/m]:', 'se3_rel_rotation_2_0', 'SE(3) rel. rotation (2m) [$\\frac{deg}{m}$]'),
           ('SIM3_ATE_RMSE[cm]:', 'sim3_ate_rmse', 'Sim(3) ATE RMSE [cm]'),
           ('SIM3_REL_TRANSLATION_0.5M[%]:', 'sim3_rel_translation_0_5', 'Sim(3) rel. translation (0.5m) [\\%]'),
           ('SIM3_REL_ROTATION_0.5M[deg/m]:', 'sim3_rel_rotation_0_5', 'Sim(3) rel. rotation (0.5m) [$\\frac{deg}{m}$]'),
           ('SIM3_REL_TRANSLATION_1M[%]:', 'sim3_rel_translation_1_0', 'Sim(3) rel. translation (1m) [\\%]'),
           ('SIM3_REL_ROTATION_1M[deg/m]:', 'sim3_rel_rotation_1_0', 'Sim(3) rel. rotation (1m) [$\\frac{deg}{m}$]'),
           ('SIM3_REL_TRANSLATION_1.5M[%]:', 'sim3_rel_translation_1_5', 'Sim(3) rel. translation (1.5m) [\\%]'),
           ('SIM3_REL_ROTATION_1.5M[deg/m]:', 'sim3_rel_rotation_1_5', 'Sim(3) rel. rotation (1.5m) [$\\frac{deg}{m}$]'),
           ('SIM3_REL_TRANSLATION_2M[%]:', 'sim3_rel_translation_2_0', 'Sim(3) rel. translation (2m) [\\%]'),
           ('SIM3_REL_ROTATION_2M[deg/m]:', 'sim3_rel_rotation_2_0', 'Sim(3) rel. rotation (2m) [$\\frac{deg}{m}$]')]


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

def ReadResult(text):
    result = dict()
    
    lines = text.split('\n')
    for line in lines:
        if (line.startswith('Could not open trajectory file:') or
            line.startswith('ComputePosesAtTimestamps(): Input poses are empty!')):
            return None
        
        words = line.rstrip('\n').split()
        
        if len(words) == 0:
            continue
        elif words[0] == 'WARNING:':
            continue
        
        word_identified = False
        for (metric_name, _, _) in metrics:
            if words[0] == metric_name:
                word_identified = True
                result[words[0]] = float(words[1])
                if result[words[0]] != result[words[0]]:  # NaN
                    return None
                break
        
        if not word_identified:
            print('Unknown word at start of line in ' + result_path + ': ' + words[0])
    
    return result

if __name__ == '__main__':
    # Check that the paths have been set
    if len(evaluation_program_path) == 0 or len(eth3d_datasets_path) == 0:
        print('Before using this script, please set evaluation_program_path in this script to the path of the' +
              ' evaluation program executable, and set eth3d_datasets_path' +
              ' in this script to the path of the ETH3D SLAM datasets on your system.')
        sys.exit(1)
    
    # Print the paths such that the user is aware of them
    print('Datasets path: ' + eth3d_datasets_path)
    print('Evaluation program: ' + evaluation_program_path)
    
    if len(sys.argv) < 3:
        print('Usage:')
        print('python evaluate_eth3d_slam_results.py result_name_1 [result_name_2 ...]')
        print('')
        sys.exit(1)
    
    # Parse arguments
    result_names = sys.argv[1:]
    
    # Create individual evaluations of each given result
    training_path = os.path.join(eth3d_datasets_path, 'training')
    results = {}  # [result_name][dataset_name][metric_tag] -> metric_value (float)
    dataset_count = 0
    
    for result_name in result_names:
        evaluation_path = os.path.join(eth3d_datasets_path, 'evaluations', result_name)
        if os.path.isdir(evaluation_path):
            shutil.rmtree(evaluation_path)
        MakeDirsExistOk(evaluation_path)
        
        results[result_name] = {}
        dataset_count = 0
        count_nan = 0
        
        for dataset_name in os.listdir(training_path):
            dataset_path = os.path.join(training_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            
            results_folder = os.path.join(dataset_path, 'results')
            if not os.path.isdir(results_folder):
                continue
            
            dataset_count += 1
            result_trajectory_path = os.path.join(results_folder, result_name + '.txt')
            result_runtime_path = os.path.join(results_folder, result_name + '_runtime.txt')
            
            if not os.path.isfile(result_trajectory_path) or not os.path.isfile(result_runtime_path):
                print('Warning: No complete result found for result ' + result_name + ' for dataset: ' + dataset_name + ' (path: ' + dataset_path + ')')
                continue
            
            results[result_name][dataset_name] = {}
            
            dataset_evaluation_path = os.path.join(evaluation_path, dataset_name)
            MakeDirsExistOk(dataset_evaluation_path)
            
            # Run the evaluation program
            is_sfm_dataset = dataset_name.startswith('sfm_')
            max_interpolation_timespan = (1. / 11.25) if is_sfm_dataset else (1. / 75.)
            
            # Yaw + translation
            if not is_sfm_dataset:
                TODO = 1  # TODO
            
            # SE(3) and Sim(3)
            call = [evaluation_program_path,
                    os.path.join(dataset_path, 'groundtruth.txt'),
                    result_trajectory_path,
                    os.path.join(dataset_path, 'rgb.txt'),
                    '--max_interpolation_timespan',
                    str(max_interpolation_timespan)]
            print('Running: ' + ' '.join(call))
            proc = subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_evaluation_path)
            stdout, stderr = proc.communicate()
            return_code = proc.returncode
            del proc
            if return_code == 0:
                text_output = stdout.decode("utf-8")
                with open(os.path.join(dataset_evaluation_path, 'result.txt'), 'wb') as outfile:
                    outfile.write(StrToBytes(text_output))
                tmp = ReadResult(text_output)
                if tmp is not None:
                    results[result_name][dataset_name].update(tmp)
                else:
                    count_nan = count_nan + 1
            else:
                print('Evaluation call failed (return code: ' + str(return_code) + ')')
            
            ## Sim(3) only
            #call = [evaluation_program_path,
                    #os.path.join(dataset_path, 'groundtruth.txt'),
                    #result_trajectory_path,
                    #os.path.join(dataset_path, 'rgb.txt'),
                    #'--sim3',
                    #'--max_interpolation_timespan',
                    #str(max_interpolation_timespan)]
            #print('Running: ' + ' '.join(call))
            #proc = subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=dataset_evaluation_path)
            #stdout, stderr = proc.communicate()
            #return_code = proc.returncode
            #del proc
            #if return_code == 0:
                #text_output = stdout.decode("utf-8")
                #with open(os.path.join(dataset_evaluation_path, 'sim3_result.txt'), 'wb') as outfile:
                    #outfile.write(StrToBytes(text_output))
                #results[result_name][dataset_name].update(ReadResult(text_output))
            #else:
                #print('Evaluation call failed (return code: ' + str(return_code) + ')')
        print("*"*10)
        print("{} / {} NaN founds in {}".format(count_nan, dataset_count ,result_name))
        print("*"*10)
    # If multiple results are given, create a cumulative comparison plot
    # X axis: error value
    # Y axis: number of runs that have less or equal error
    if len(result_names) > 1:
        # Use Latex default font for plots
        matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        # matplotlib.rc('text', usetex=True) #jzm: get error on this
        
        today = datetime.datetime.now()
        comparison_path = os.path.join(eth3d_datasets_path,
                                       'comparisons',
                                       today.strftime('%Y-%m-%d_%H-%M-%S') + '_' + '_'.join(result_names))
        MakeDirsExistOk(comparison_path)
        
        for (metric_tag, metric_filename, metric_latex) in metrics:
            fig, axes = plt.subplots(1, 1, figsize=(3 / 1.5, 1.865 / 1.5))
            
            for result_name in result_names:
                sorted_results = []
                for dataset_name in results[result_name]:
                    if (metric_tag in results[result_name][dataset_name] and
                        results[result_name][dataset_name][metric_tag] == results[result_name][dataset_name][metric_tag]):  # NaN check
                        sorted_results.append(results[result_name][dataset_name][metric_tag])
                sorted_results.sort()
                
                current_y = 0
                plot_x_values = []
                plot_y_values = []
                for result in sorted_results:
                    plot_x_values.append(result)
                    plot_y_values.append(current_y)
                    current_y += 1
                    plot_x_values.append(result)
                    plot_y_values.append(current_y)
                plot_x_values.append(999999)  # something to the right of the plot
                plot_y_values.append(current_y)
                
                method_results = [0 for _ in plot_x_values]
                
                axes.plot(plot_x_values,
                          plot_y_values,
                          label=result_name)
            
            #plt.title('Title')
            plt.ylabel('\\#successful runs')
            plt.xlabel(metric_latex)
            
            ## Try to counteract a matplotlib problem with overlapping tick labels
            #if len(axes.xaxis.get_ticklabels()) >= 9:
              #for label in axes.xaxis.get_ticklabels()[::2]:
                #label.set_visible(False)
            
            lgd = axes.legend(fancybox=False, borderaxespad=0,  # borderpad=1.0,
                              loc='upper left', bbox_to_anchor=[1.05, 1]) #, ncol=2)
            
            plt.xlim(0, 8)
            plt.ylim(0, dataset_count)
            fig.savefig(os.path.join(comparison_path, metric_filename + '_cumulative_error.pdf'), dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
            
            plt.xlim(0, 2.0)
            plt.ylim(0, dataset_count)
            fig.savefig(os.path.join(comparison_path, metric_filename + '_cumulative_error_closeup.pdf'), dpi=300, bbox_inches='tight', bbox_extra_artists=(lgd,))
            
            del fig
