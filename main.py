import csv
import gc
import glob
import os
import sys
import time

from Dataset import create_dataset
from Killer import kill
from Model import train_model
from MutFinder import find
from MutationCreater import create_mut


def train(file_name, dataset_size, num_muts, model_file, number_muts_per, end_mut):

    """
       Trains and tests model
       :param str file_name:            Name of the spec FSM file
       :param int dataset_size:         Size of the dataset
       :param int num_muts:             Number of mutants that will attempt to be killed
       :param str model_file:           Name of the models save file name
       :param int number_muts_per:       Number of mutations per mutant
       :param boolean end_mut:          If true, mutations will only happen on the final state
       :return float modelTime:        Time in seconds that this function took to complete  (Time model took to train and kill mutants)
       :return int lines:              Number of lines in the FSM
       :return int model_not_kill_count:  Number of mutants not killed by model
       :return float[][] con:          2D array of the confusion matrix for the model
       :return int total_states:        Number of states in the FSM
       :return int total_inputs:        Number of inputs in the FSM
       :return int total_outputs:        number of outputs in the FSM
       """
    t0 = time.time()
    total_states, lines, total_inputs, total_outputs = create_dataset(dataset_size, file_name, number_muts_per, end_mut)
    print(f'Total States: {total_states}')
    print(f'Total Inputs: {total_inputs}')
    print(f'Total Outputs: {total_outputs}')
    gc.collect()
    t1 = time.time()
    print(t1 - t0)

    con = train_model(1e-3, lines, model_file)
    # con = []
    gc.collect()
    create_mut(num_muts, file_name, number_muts_per, end_mut)

    gc.collect()
    model_not_kill_count = find(lines, file_name, 100, True, 0, model_file)

    gc.collect()
    t2 = time.time()
    return t2 - t0, lines, model_not_kill_count, con, total_states, total_inputs, total_outputs


file_name = f"FSMs/0.txt"

numMutsStart = 500  # This is 10x for the first one

modelFile = "First"
print(sys.argv)
if len(sys.argv) >= 3:
    file_name = sys.argv[1]
    mode = sys.argv[2]
    numberMutsPer = int(sys.argv[3])
    if sys.argv[4] == 'False':

        endMut = False
    elif sys.argv[4] == 'True':
        endMut = True
    else:
        print("Error")
        exit()
else:
    file_name = "Example FSM.txt"
    mode = "1"
    numberMutsPer = 1
    endMut = False
print(file_name)

print(file_name)
files = glob.glob('mut/*')
for f in files:
    os.remove(f)
files = glob.glob('mutFinal/*')
for f in files:
    os.remove(f)
files = glob.glob('Dataset/Correct/*')
for f in files:
    os.remove(f)
files = glob.glob('Dataset/Wrong/*')
for f in files:
    os.remove(f)
files = glob.glob('Evaluation/Validation/*')
for f in files:
    os.remove(f)

fileCSV = "Results/Results"
if not os.path.exists(f'{fileCSV}.csv'):
    with open(f'{fileCSV}.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["Number of states", "Number of inputs", "Number of Outputs", "Number of Mutations per Mutant",
                         "Number of Muts",
                         "Model time", "False Negative", "False Positive", "True Negative", "True Positive",
                         "Model Not Killed Count",
                         "Depth time", "Depth Not Killed Count", "Breath time", "Breath Not Killed Count"])
        f.close()
"""
Each mode is a difference experiment

Mode 1:     Model method
Mode 2:     Depth first search method
Mode 3:     Breath first search method

This need to be called in order so that logging is correct
"""
numMuts = 10000

print(f'end mut: {endMut}')
if mode == "1":
    modelTime, lines, modelNotKillCount, con, totalStates, totalInputs, totalOutputs = train(file_name, 80000, numMuts,
                                                                                             modelFile, numberMutsPer,
                                                                                             endMut)
    with open(f'{fileCSV}.csv', 'a') as f:
        writer = csv.writer(f)
        f.write(
            f'{totalStates}, {totalInputs}, {totalOutputs}, {numberMutsPer}, {numMuts}, {modelTime}, {con[1][1]}, {con[0][1]}, {con[1][0]}, {con[0][0]}, {modelNotKillCount}, ')
        f.close()
elif mode == "2":
    totalStates, lines, totalInputs, totalOutputs = create_dataset(10, file_name, numberMutsPer, endMut)
    t6 = time.time()
    create_mut(numMuts, file_name, numberMutsPer, endMut)
    gc.collect()
    find(lines, file_name, 100, False, 0, file_name)
    gc.collect()
    non_num_seq1, non_seqs1, non_seqs_short1, non_amount_killed1, depth_not_kill_count = kill(numMuts, file_name, 1)  # Depth
    gc.collect()
    t7 = time.time()
    depthTime = t7 - t6
    with open(f'{fileCSV}.csv', 'a') as f:
        writer = csv.writer(f)
        f.write(f'{depthTime}, {depth_not_kill_count}\n')
        f.close()
elif mode == "3":
    totalStates, lines, totalInputs, totalOutputs = create_dataset(10, file_name, numberMutsPer, endMut)
    t8 = time.time()
    create_mut(numMuts, file_name, numberMutsPer, endMut)
    gc.collect()
    find(lines, file_name, 100, False, 0, file_name)
    gc.collect()
    non_num_seq2, non_seqs2, non_seqs_short2, non_amount_killed2, breath_not_kill_count = kill(numMuts, file_name, 2)  # breath
    gc.collect()
    breathTime = time.time() - t8
    with open(f'{fileCSV}.csv', 'a') as f:
        writer = csv.writer(f)
        f.write(f' {breathTime}, {breath_not_kill_count}\n')
        f.close()
else:
    print("Invalid Mode")
