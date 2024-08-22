import sys
import time

import enlighten
import numpy as np

from FSMsim import sim

sys.setrecursionlimit(1000000)
label = {}

def read_fsm(file_path, inputs):
    """
    Reads FSM file and returns dictionary
    :param file_path: FSM file path
    :param inputs: Blank set of inputs
    :return: FSM as a dictionary with a complete set of inputs
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    fsm = {}
    for line in lines:
        elements = line.strip().split(' ')
        if len(elements) != 4:
            raise ValueError('Each line should have 4 characters separated by a space')

        startstate, endstate, input1, output = elements[0], elements[1], elements[2], elements[3]
        inputs.add(input1)
        if startstate not in fsm:
            fsm[startstate] = {}

        if endstate not in fsm[startstate]:
            fsm[startstate][endstate] = []
        if len(fsm[startstate][endstate]) > 0:
            i = 0
            while i < round(len(fsm[startstate][endstate]) / 2) + 1:

                try:
                    if len(fsm[startstate][endstate]) == 0:
                        fsm[startstate][endstate] = []
                except:
                    fsm[startstate][endstate] = []

                i += 1

        fsm[startstate][endstate].append(str(input1))
        fsm[startstate][endstate].append(str(output))
    return fsm, inputs


def get_state(spec, current_state, current_input):
    """
    :param spec: Dictionary of the specification FSM
    :param current_state: Current state of the FSM
    :param current_input: Current input to find next state for
    :return: Next state from the current state and current input
    """

    for next_state in spec[current_state]:
        for i in range(0, len(spec[current_state][next_state])):
            if spec[current_state][next_state][i] == current_input:
                return next_state
            i += 1

    return -1


def get_next_state(spec, current_state, current_input):
    """

    :param spec: Dictionary of the specification FSM
    :param current_state: Current state of the FSM
    :param current_input: Current input to find next state for
    :return: Next state from the current state and current input
    """
    for next_state in spec[current_state]:

        if current_input in spec[current_state][next_state]:
            return next_state

    return -1


def depth(spec, mut, mut_current_state, spec_current_state, seq, used, inputs, input_index, is_root, reverse, start_time):
    """
    Depth first search.
    :param spec: Specification FSM
    :param mut: Mutation FSM
    :param mut_current_state: Current mutation state
    :param spec_current_state: Current spec mutation state
    :param seq: Output sequence
    :param used: Array of the states that have been used
    :param inputs: Array of inputs
    :param input_index: Current input index
    :param is_root: Boolean if the current node is the root node
    :param reverse: If going back up the tree
    :param start_time: The time at start of search
    :return: The input sequence with a boolean of whether it found a difference
    """
    time_spent = time.time() - start_time
    if time_spent >= 10:
        return seq, False
    dig = False
    while not dig:
        if not spec_current_state.isdigit():
            spec_current_state = spec_current_state[:-1]
        elif not mut_current_state.isdigit():
            mut_current_state = mut_current_state[:-1]
        else:
            dig = True

    tmp1 = get_state(spec, spec_current_state, inputs[input_index])
    tmp2 = get_state(mut, mut_current_state, inputs[input_index])
    if tmp1 == -1 or tmp2 == -1:
        if input_index + 1 >= len(inputs):
            used.append([spec_current_state, mut_current_state])
        if tmp1 == -1 and tmp2 == -1:
            return seq, False
        else:
            print(inputs[input_index])
            seq.append(inputs[input_index])
            return seq, True

    if len(spec[spec_current_state][tmp1]) > len(mut[mut_current_state][tmp2]):
        length = len(spec[spec_current_state][tmp1])
    else:
        length = len(mut[mut_current_state][tmp2])

    a1 = 0
    a2 = 0
    for k in range(0, int(length / 2)):
        if spec[spec_current_state][tmp1][(k * 2)] == inputs[input_index]:
            a1 = k
            break
    for k in range(0, int(length / 2)):
        if mut[mut_current_state][tmp2][k * 2] == inputs[input_index]:
            a2 = k
            break
    if spec[spec_current_state][tmp1][1 + (a1 * 2)] == mut[mut_current_state][tmp2][1 + (a2 * 2)]:

        if [spec_current_state, mut_current_state] in used and reverse == False:
            return seq, False

        if is_root:

            for input_index_tmp in range(0, len(inputs)):

                seq, done = depth(spec, mut, mut_current_state, spec_current_state, seq, used, inputs, input_index_tmp, False,
                                  False, start_time)

                if done:
                    return seq, True
            return seq, False
        used.append([spec_current_state, mut_current_state])

        old_mut = mut_current_state
        old_spec = spec_current_state
        spec_current_state = get_next_state(spec, spec_current_state, inputs[input_index])
        mut_current_state = get_next_state(mut, mut_current_state, inputs[input_index])

        seq, done = depth(spec, mut, mut_current_state, spec_current_state, seq, used, inputs, 0, False, False, start_time)
        input_index_tmp = 0
        if done == False and reverse == False:
            while input_index_tmp + 1 < len(inputs):
                input_index_tmp += 1

                seq, done = depth(spec, mut, old_mut, old_spec, seq, used, inputs, input_index_tmp, False, True, start_time)

                if done:
                    return seq, True

            return seq, False

        elif done:

            seq.append(inputs[input_index])
            return seq, True
        else:
            return seq, False

    else:

        seq.append(inputs[input_index])
        return seq, True


def breath(spec, mut, mut_next_state, spec_next_state, seq, used, inputs, input_index, is_root, past_mut, past_spec,
           reverse, current_depth, max_depth, past_depth, start_time, count):
    """
    :param spec: Dictionary of the specification FSM
    :param mut: Dictionary of the mutant FSM
    :param mut_next_state: Next state for the mutant FSM
    :param spec_next_state: Next state for the specification FSM
    :param seq: Sequences to find the error
    :param used: This sorts the states that have been explored
    :param inputs: Array of inputs that the FSM has
    :param input_index: Index of current input in inputs array
    :param is_root: Tells if the current node is the root of the FSM
    :param past_mut: The past state for the mutant FSM
    :param past_spec: The past state for the specification FSM
    :param reverse: Tells if the search is going backwards
    :param current_depth: Current current_depth of the search
    :param max_depth: Maximum current_depth for this iteration
    :param past_depth: Past maximum current_depth
    :param start_time: Time the search started
    :param count: Current count
    :return: The sequence needed to find difference, whether it found a difference, the states visited, and the count.
    """
    count += 1


    time_spent = time.time() - start_time
    if time_spent >= 10:
        return seq, False, used, count

    dig = False
    while not dig:
        if not spec_next_state.isdigit():
            spec_next_state = spec_next_state[:-1]
        elif not mut_next_state.isdigit():
            mut_next_state = mut_next_state[:-1]
        else:
            dig = True

    spec_current_state = past_spec
    mut_current_state = past_mut
    if is_root:

        old_mut = mut_current_state
        old_spec = spec_current_state
        used.append([old_spec, old_mut])

        for max_depth in range(0, 10000000000000):


            for input_index_tmp in range(0, len(inputs)):

                time_spent = time.time() - start_time
                if time_spent >= 10:
                    return seq, False, used, count
                spec_current_state = get_next_state(spec, old_spec, inputs[input_index_tmp])
                mut_current_state = get_next_state(mut, old_mut, inputs[input_index_tmp])

                seq, done, used, count = breath(spec, mut, mut_current_state, spec_current_state, seq, used, inputs,
                                                input_index_tmp, False, old_mut, old_spec, False, current_depth, max_depth,
                                                max_depth - 1, start_time, count)

                if done:
                    seq.append(inputs[input_index_tmp])
                    return seq, True, used, count
        return seq, False, used, count
    if spec_next_state == -1 or mut_next_state == -1:
        if input_index + 1 >= len(inputs):
            used.append([spec_current_state, mut_current_state])
        if spec_next_state == -1 and mut_next_state == -1:
            return seq, False, used, count
        else:
            print(inputs[input_index])
            seq.append(inputs[input_index])
            return seq, False, used, count
    done = False

    if len(spec[spec_current_state][spec_next_state]) > len(mut[mut_current_state][mut_next_state]):
        length = len(spec[spec_current_state][spec_next_state])
    else:
        length = len(mut[mut_current_state][mut_next_state])

    a1 = 0
    a2 = 0
    for k in range(0, int(length / 2)):
        if spec[spec_current_state][spec_next_state][(k * 2)] == inputs[input_index]:
            a1 = k
            break
    for k in range(0, int(length / 2)):
        if mut[mut_current_state][mut_next_state][k * 2] == inputs[input_index]:
            a2 = k
            break

    if spec_current_state == 3:
        print("Hold")
    if spec[spec_current_state][spec_next_state][1 + (a1 * 2)] == mut[mut_current_state][mut_next_state][1 + (a2 * 2)]:

        if is_root == False and max_depth == current_depth:
            return seq, False, used, count

        if current_depth >= max_depth:
            return seq, False, used, count

        mut_current_state = mut_next_state
        spec_current_state = spec_next_state

        if done == False and reverse == False:
            for input_index_tmp in range(0, len(inputs)):

                time_spent = time.time() - start_time
                if time_spent >= 10:
                    return seq, False, used, count
                spec_next_state = get_next_state(spec, spec_current_state, inputs[input_index_tmp])
                mut_next_state = get_next_state(mut, mut_current_state, inputs[input_index_tmp])

                seq, done, used, count = breath(spec, mut, mut_next_state, spec_next_state, seq, used, inputs,
                                                input_index_tmp, False, mut_current_state, spec_current_state, False,
                                                current_depth + 1, max_depth, past_depth, start_time, count)

                if done:
                    seq.append(inputs[input_index_tmp])
                    return seq, True, used, count

            return seq, False, used, count

        elif done:

            seq.append(inputs[input_index])
            return seq, True, used, count
        else:
            return seq, False, used, count
    else:

        seq.append(inputs[input_index])
        return seq, True, used, count


def kill(muts, file_name, mode):
    """
    This is the main function for calling killer code.
    :param muts: Number of mutations
    :param file_name: Name of FSM file
    :param mode: Which search method mode is to be used (1 current_depth) (2 breath)
    :return: The number of sequence after prefixes removed, set of all sequences tested, a set of the sequences with
             prefixes removed, number of mutants killed, number of mutants not killed.
    """
    if mode == 1:
        print("Depth")
    else:
        print("Breath")
    not_killed_count = 0
    print("Kill")
    inputs = set()
    spec, inputs = read_fsm(file_name, inputs)

    seq_set = {}

    not_done = np.arange(0, muts)
    manager = enlighten.get_manager()
    counter1 = manager.counter(total=muts, desc='Amount killed', unit='FSMs', leave=False)
    count_done = 0
    count1 = 0

    while len(not_done) > 0:

        count1 += 1
        old = len(not_done)

        inputs = set()

        mut, inputs = read_fsm(f'mutFinal/{not_done[0]}.txt', inputs)

        inputs = list(inputs)
        inputs.sort()
        list_spec = list(spec)
        list_mut = list(mut)
        used = []

        if mode == 1:
            start_time = time.time()
            seq, done = depth(spec, mut, list_mut[0], list_spec[0], [], used, inputs, 0, True, False, start_time)
        else:
            start_time = time.time()
            seq, done, used, count2 = breath(spec, mut, list_mut[0], list_spec[0], [], used, inputs, 0, True,
                                             list_mut[0], list_spec[0], False, 0, 0, 0, start_time, 0)

        seq.reverse()

        str_seq = ''.join(seq)
        spec_out = sim([file_name, seq])
        mut_out = sim([f'mutFinal/{not_done[0]}.txt', seq])

        if str_seq == "":
            print(f'ERROR.. {not_done[0]}')
        elif spec_out == mut_out:
            print(f'Same ERROR.. {not_done[0]}')
        #
        if not done:
            print("ERROR")
            not_killed_count += 1
        kill_count = 0

        not_done = np.delete(not_done, 0)
        counter2 = manager.counter(total=len(not_done), desc='Check amount killed', unit='FSMs', leave=False)
        mut_outs = []

        for i in range(0, len(not_done)):
            mut_outs.append([sim((f'mutFinal/{not_done[i]}.txt', seq)), not_done[i]])

        del seq

        for i in range(0, len(mut_outs)):

            if spec_out != mut_outs[i][0]:

                kill_count += 1

                if mut_outs[i][1] in not_done:
                    not_done = np.delete(not_done, np.where(not_done == mut_outs[i][1])[0][0])

            counter2.update()

        counter2.close(clear=True)
        seq_set[str_seq] = {}
        seq_set[str_seq]["KillCount"] = []
        seq_set[str_seq]["KillCount"].append(str(((old - len(not_done)) / old) * 100) + "%")
        counter1.update(old - len(not_done))

    count_done += muts - len(not_done)
    del not_done
    tmp = seq_set.keys()
    final_seq = []
    for a in tmp:
        final_seq.append(a)
    tmp3 = final_seq.copy()
    final_seq.reverse()

    for a in final_seq:
        for i in tmp3:
            if str(a).startswith(str(i)) == True and a != i:
                final_seq.remove(i)
                tmp3.remove(i)
                break

    print(f'Number of mutants tested: {count1}')
    print(f'Number of sequences needed: {len(final_seq)}')

    print(len(seq_set))
    return len(final_seq), seq_set, final_seq, count_done, not_killed_count
