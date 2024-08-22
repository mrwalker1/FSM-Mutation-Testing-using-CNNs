import multiprocessing
import random
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from alive_progress import alive_bar


def save(args):
    """
       This creates the mutants and saves the images
       :param args: This is an array of the inputs for this function(
        numpyArray master_np:     FSM as numpy array
        int i:                   The current iteration
        int outName:             Name of output directory
        int count:             Number of rows in FSM
        str[] states:            Array of the states in the FSM
        str file_dir:             Dataset output directory
        str[] outputs:            Array of the outputs in the FSM
        int number_muts_per:       Number of mutations per mutant
        boolean end_mut:           If the mutation will happen only at the last state)

       """
    master_np, i, count, states, file_dir, outputs, num_muts_per, end_mut =args
     #This makes the trans dataset
    final_check = False
     
    while not final_check:

        sel = i%4

        if sel < 2:

            img_out = master_np.copy()

            for ca in range(0,num_muts_per):
                check = False
                tmp = img_out.copy()
            
                while not check:
                    
                    if end_mut:
                        start_pos = count-len(outputs)
                        
                        a = random.randrange(start_pos,count)
                    else:
                        a = random.randrange(0,count)
                   
                    r = random.randrange(0,len(states))
                    

                    b = 1
                    col = states[r]
                    img_out[a][b] = col
                    if img_out[a][b][0] != tmp[a][b][0] or img_out[a][b][1] != tmp[a][b][1] or img_out[a][b][2] != tmp[a][b][2]:
                        check = True
            img_out = np.array(img_out)
            im = Image.fromarray(img_out.astype(np.uint8)).resize((4, count), Image.NEAREST)
            im.save(file_dir+"Validation/"+str(i)+".png")
            
        else:

            img_out = master_np.copy()

            for ca in range(0,num_muts_per):
                check = False
                tmp = img_out.copy()
                
                while not check:
                    
                    if end_mut:
                        start_pos = count-len(outputs)
                        
                        a = random.randrange(start_pos,count)
                    else:
                        a = random.randrange(0,count)


                    b =3
                    r = random.randrange(0,len(outputs))
                    
                    col = outputs[r]
                        
                    img_out[a][b] = col
                    if img_out[a][b][0] != tmp[a][b][0] or img_out[a][b][1] != tmp[a][b][1] or img_out[a][b][2] != tmp[a][b][2]:
                    
                        check = True

            img_out = np.array(img_out)
            
            im = Image.fromarray(img_out.astype(np.uint8)).resize((4, count), Image.NEAREST)
            im.save(file_dir+"Validation/"+str(i)+".png")
        if (img_out == master_np).all():
            print("Identical")
            
        else:
            final_check = True
    del img_out
    del im, outputs
    del a,b, master_np, i, count, file_dir
def create_mut(itr, file_name, num_muts_per, end_mut):
    """
    This creates a set amount of mutants
    :param itr: Number of mutants to be created
    :param file_name: Name of output directory
    :param num_muts_per: Number of mutations per mutant
    :param end_mut: If the mutation will happen only at the last state
    """
    Path("Evaluation/Validation").mkdir(parents=True, exist_ok=True)
    im = iio.imread('colour master.png')
    print(im[0][0])
    file_dir = "Evaluation/"
    file1 = open(f'{file_name}', 'r')
    lines = file1.readlines()
    main = []
    count = 0

    list_a = set()
    list_b = set()
    list_s = set()
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

    c = im[0]#16384 is the number of states in the fsm
    print(main.shape[0])
    print(main.shape[0]*4)
    print(len(c))
    master = []
    count1 = len(list_s)
    colour = c
    states =colour[(len(list_a)+len(list_b))+1:len(list_a)+len(list_b)+(len(list_s))+1][:]#this splits off the number colours states from the input colours
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

                        amount = int(hold)
                        tmp1.append(states[amount-1][a])
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
                        tmp1.append(c[count1][a])
                tmp.append(tmp1)
            master.append(tmp)
            bar()
    master_np = np.array(master)
    img_out = master.copy()
    img_out = np.array(img_out)


    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(save, [(master_np, i, count, states, file_dir, outputs, num_muts_per, end_mut) for i in range(0, itr)])
    del pool, img_out,im, master_np,tmp,tmp1,states, inputs,outputs
    print("Mutants made")
