import multiprocessing
import random
import time
from pathlib import Path

import enlighten
import matplotlib as mpl
import numpy as np
import torch.multiprocessing as mp
from PIL import Image
from alive_progress import alive_bar

manager = enlighten.get_manager()
Path("Dataset/Correct").mkdir(parents=True, exist_ok=True)
Path("Dataset/Wrong").mkdir(parents=True, exist_ok=True)
start_time = time.time()

    
def non_mut(arg):
    """
    This functions creates FSMs that are not mutations of the specification machine.
    :param arg: This is an array of the following inputs (master_np: FSM as array, out_name: The name of the output file,
    count: This is the number of rows, file_dir: The output directory, number_muts_per: Number of mutations per FSM,
    end_mut: boolean if true the mutations will only happen on the final state)
    """
    master_np, out_name, count, file_dir, number_muts_per, end_mut = arg
    final_check = False
     
    while not final_check:
        img_out = master_np.copy()
        for i in range(0,number_muts_per):
            check = False
            tmp = img_out.copy()
            while not check:

                a = random.randrange(0,count)
                b = random.randrange(0,count)
                
                img_out[a] = tmp[b].copy()
                img_out[b] = tmp[a].copy()

                if np.any(img_out[a] != tmp[a]):
                    check = True
        img_out = np.array(img_out)

        im = Image.fromarray(img_out.astype(np.uint8))
        im.save(file_dir+"Correct/"+str(out_name)+".png")
        if (img_out == master_np).all():
            print("Identical")
            
        else:
            final_check = True
    del im
    del img_out
    del a,b, master_np, i, count, file_dir
def mut_in(arg):
    """
        This functions creates mutants of the specification machine on the output state.
        :param arg: This is an array of the following inputs (master_np: FSM as array, out_name: The name of the output file,
        count: This is the number of rows, file_dir: The output directory, number_muts_per: Number of mutations per FSM,
        end_mut: boolean if true the mutations will only happen on the final state)
        """
    master_np, out_name, count, file_dir, states, itr, number_muts_per, end_mut = arg
    final_check = False
     
    while not final_check:
        img_out = master_np.copy()
        for i in range(0,number_muts_per):
            check = False
            tmp = img_out.copy()
        
            while not check:
                a = random.randrange(0,count)

                r = random.randrange(0,len(states))

                b = 1
                col = states[r]
                img_out[a][b] = col
                if img_out[a][b][0] != tmp[a][b][0] or img_out[a][b][1] != tmp[a][b][1] or img_out[a][b][2] != tmp[a][b][2] :
                    check = True
        img_out = np.array(img_out)
        

        im = Image.fromarray(img_out.astype(np.uint8))
        im.save(file_dir+"Wrong/"+str(out_name)+".png")
        if (img_out == master_np).all():
            print("Identical")
            
        else:
            final_check = True
    del im
    del img_out
    del a,b, master_np, i, count, file_dir, itr, check

def mut_out(arg):
    """
    This functions creates mutants of the specification machine on the output.
    :param arg: This is an array of the following inputs (master_np: FSM as array, out_name: The name of the output file,
    count: This is the number of rows, file_dir: The output directory, number_muts_per: Number of mutations per FSM,
    end_mut: boolean if true the mutations will only happen on the final state)
    """
    master_np, out_name, count, file_dir, inputs, itr, outputs, number_muts_per, end_mut =arg
    #print(i)
    final_check = False
     
    while not final_check:
        img_out = master_np.copy()
        for i in range(0,number_muts_per):
            check = False
            tmp = img_out.copy()
            
            while not check:
                a = random.randrange(0,count)

                b =3
                r = random.randrange(0,len(outputs))
                col = outputs[r]
                    
                img_out[a][b] = col
                if img_out[a][b][0] != tmp[a][b][0] or img_out[a][b][1] != tmp[a][b][1] or img_out[a][b][2] != tmp[a][b][2]:
                
                    check = True
        img_out = np.array(img_out)

        im = Image.fromarray(img_out.astype(np.uint8))
        im.save(file_dir+"Wrong/"+str(out_name+round(itr/2))+".png")
        if (img_out == master_np).all():
            print("Identical")
            
        else:
            final_check = True
    del im
    del img_out
    del a,b, master_np, i, count, file_dir, itr, check
    return
def generate_unique_colors_with_labels(num_colors):
    """
    This creates a list of unique colors based on the number of colors.
    :param num_colors: number of colors to generate
    :return: A list of unique colors
    """
    colors = set()  # Use a set to store unique colors
    angle_range = 360.0 / num_colors
    for i in range(num_colors):
        rgb_values = mpl.colors.hsv_to_rgb([[(i * angle_range)/360, 1.0,1.0]])
        red = np.uint8(rgb_values[0, 0] * 255)
        green = np.uint8(rgb_values[0, 1] * 255)
        blue = np.uint8(rgb_values[0, 2] * 255)
        color = (red, green, blue)
        colors.add(color)
    return list(colors)
def read_file(fsm_file):
    """
    This reads the FSM file.
    :param fsm_file: FSM file
    :return: The FSM split into its states, inputs, outputs, number of lines and the fsm as array.
    """
    count = 0
    main = []
    lines = fsm_file.readlines()
    list_a = set()
    list_b = set()
    list_s = set()
    print(f'{len(lines)},Num of lines')
    for line in lines:
        
        l = line.strip()
        s = l.split()
        list_a.add(s[2])
        list_b.add(s[3])
        list_s.add(int(s[0]))
        main.append(s)
        count += 1
    list_a = list(list_a)
    list_a.sort()
    list_b = list(list_b)
    list_b.sort()
    list_s = list(list_s)
    list_s.sort()
    main = np.array(main)
    return list_a,list_b,list_s, lines, main,count
def create_dataset(num, file_name, number_muts_per, end_mut):
    """
    This creates a dataset of specification machines and mutations
    :param num: number of rows
    :param file_name: name of the file
    :param number_muts_per: number of mutations per FSM
    :param end_mut: boolean if true the mutations will only happen on the final state
    """
    file_dir = "Dataset/"

    fsm_file = open(f'{file_name}', 'r')
    
    
    
    # Strips the newline character
    list_a,list_b,list_s, lines, main,count =read_file(fsm_file)

    colour = generate_unique_colors_with_labels(len(list_a)+len(list_b)+(len(list_s)+1))# Is creates the colours

    print(len(colour))
    master = []
    count1 = len(main)
    states =colour[(len(list_a)+len(list_b)):len(list_a)+len(list_b)+(len(list_s)+1)][:]#this splits off the number colours states from the input colours
    inputs =  colour[0:len(list_a)] # this splits of the inputs from the other colours
    outputs= colour[len(list_a):len(list_a)+len(list_b)]
    

    count = main.shape[0]
    with alive_bar(main.shape[0]) as bar:    
        for i in range(main.shape[0]):
            tmp = []
            for j in range(main.shape[1]):
                tmp1 = []
                
                if j<2:
                    hold = main[i][j]
                    for a in range(0,3):
                        #amount = int(hold[1:])
                        amount = int(hold)                                        
                        tmp1.append(states[amount][a])                        
                elif j ==2:           
                    hold = main[i][j]                 
                    for a in range(0,3):
                        tmp1.append(inputs[list_a.index(hold)][a])
                elif j ==3:                   
                    hold = main[i][j]
                    for a in range(0,3):
                        tmp1.append(outputs[list_b.index(hold)][a])
                else:
                    print("ERROR")
                    count1+=1
                    for a in range(0,3):
                        tmp1.append(colour[count1][a])
                tmp.append(tmp1)
            master.append(tmp)
            bar()
    del tmp1,tmp
    print(f'{len(lines)},Num of lines')
    master_np = np.array(master)
    img_out = master.copy()
    img_out = np.array(img_out)
    im = Image.fromarray(img_out.astype(np.uint8))
    im.save("master.png")
    del master, im

    colour = np.array(colour)
    colour = [colour]
    img_out = colour.copy()
    img_out = np.array(img_out)
    im = Image.fromarray(img_out.astype(np.uint8))
    im.save("colour master.png")
    del img_out,im

    itr = num 
    del num
    itr = round(itr/2)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(non_mut, [(master_np, i, count, file_dir, number_muts_per, end_mut) for i in range(0, round(itr))])
    del pool
    print("Non done")
    with mp.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(mut_in, [(master_np, i, count, file_dir, states, itr, number_muts_per, end_mut) for i in range(0, round(itr / 2))])
    del pool
    print("In done")
    with mp.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(mut_out, [(master_np, i, count, file_dir, inputs, itr, outputs, number_muts_per, end_mut) for i in range(0, round(itr / 2))])
    del pool
    print("Out done")

    
    print("Dataset Made")
    return max(list_s), len(lines), len(list(dict.fromkeys(list_a))), len(list(dict.fromkeys(list_b)))

