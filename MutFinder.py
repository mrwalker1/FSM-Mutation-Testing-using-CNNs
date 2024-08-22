import multiprocessing
import shutil
from multiprocessing import Value
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tensorflow as tf
from alive_progress import alive_bar

from Dataset import read_file
from Model import average_model

Path("mut").mkdir(parents=True, exist_ok=True)
Path("mutFinal").mkdir(parents=True, exist_ok=True)
    
counter = Value('i', 0)
def mut_out(args):
    #nonlocal count, lock
    """
        Coverts the images back to text files
        :param args: This is an array of the inputs for this function(Mutant image
        List of states in FSM
        List of inputs in FSM
        List of outputs in FSM
        2D array of the colours for the FSMs inputs
        2D array of the colours for the FSMs outputs
        2D array of the colours for the FSMs states
        Temp score of the mutant)
        """
    global counter
    image, check, states, inputs, outputs, list_a, list_b,list_s, tmp_score = args
    #list_a = ["a","b","c","d","e","f"]
    out = []
    x= 0
  
    for row in image:
    #if x%10 == 0:
        hold = []
        elm = np.round(row[0]*255)#this gets the element and gets it to the same value as the colour which has to be rounded as some rounding happens earlier in the process making them not identical
        pos= np.where(np.all(states == elm, axis=-1))[0][0]#This finds the position of that colour
        
        hold.append(list_s[pos])#This adds that as the number
        
        elm = np.round(row[1]*255)
        pos= np.where(np.all(states == elm, axis=-1))[0][0]
        
        hold.append(list_s[pos])
        
        elm = np.round(row[2]*255)
        pos= np.where(np.all(inputs == elm, axis=-1))[0][0]
        
        hold.append(list_a[pos])#This adds it as the input letter
        
        elm = np.round(row[3]*255)
        pos= np.where(np.all(outputs == elm, axis=-1))[0][0]
    
        hold.append(list_b[pos])
        out.append(hold)
        x+=1
    out = np.array(out)

    #while check == False:
    with counter.get_lock():
        np.savetxt("mut/"+str(counter.value)+'.txt', out, delimiter=' ',  fmt='%s')
        counter.value += 1

    
    return 0
def init(args):
    """ store the counter for later use """
    global counter
    counter = args
def find(height, filename, batch_size, with_model, pred, model_file):
    """
    :param height: height of the FSM
    :param filename: name of the FSM file
    :param batch_size: batch size
    :param with_model: if the model is to be used or not
    :param pred: Which model output to look at
    :param model_file: path to the model
    """
   
    not_killed_count = 0

    image_shape = (height, 4)

    datagen_kwargs = dict(rescale=1./255,  validation_split=.40, fill_mode='nearest')
    



    
    learning_rate = 0.000001
    print(height)
    model = average_model(height)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['acc',tf.keras.metrics.Precision(), tf.keras.metrics.BinaryAccuracy(
    name='binary_accuracy', dtype=None, threshold=0.5
    )])
    if with_model:
        model.load_weights(f'models/{model_file}.ckpt').expect_partial()
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    eval1 = eval_datagen.flow_from_directory(
        "Evaluation",
        
        
        target_size=image_shape,
        batch_size= batch_size,
        
    )

    tmp_score = 0
    print(eval1.samples)
    image_batch, label_batch =[],[]
    for image_batch,label_batch in eval1:

        break
    print(image_batch.shape, label_batch.shape)
   

    im = iio.imread('colour master.png')
    file1 = open(f'{filename}', 'r')
    list_a,list_b,list_s, lines, main,count =read_file(file1)
    colour = im[0]

    states =colour[(len(list_a)+len(list_b))+1:len(list_a)+len(list_b)+(len(list_s))+1][:]#this splits off the number colours states from the input colours
    inputs =  colour[0:len(list_a)] # this splits of the inputs from the other colours
    outputs= colour[len(list_a):len(list_a)+len(list_b)]
    scores = []
    r= np.empty((0,2))
    i = 0
    counter = Value('i', 0)
    count = 0
    if not with_model:
        for a in range(0, eval1.samples):
            #print(count)
            
            r =np.vstack([r, [count,count]])
            count+=1
    with alive_bar(len(eval1)) as bar:
        for image_batch,label_batch in eval1:
            
            if i >=eval1.samples/batch_size:
                break
            #t = np.expand_dims(image_batch, axis=0)

            if with_model:
                r= np.vstack([r, model.predict(image_batch, verbose=0)])
                
                
                
            #print(multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initargs = (counter, ),initializer = init) as pool:
                pool.map(mut_out, [(image, False, states, inputs, outputs, list_a, list_b, list_s, tmp_score) for image in image_batch])
            del pool
            del image_batch
            
            i+=1
            bar()
        print(len(r))
        for a in range(0,len(r)):
            #print(r[a])
            if r[a][0]>0.5:
                not_killed_count+=1
        print(f'Not Killed: {not_killed_count}')
    del model

    print(r.shape)
    scores.append(r[:,pred])
    scores = scores[0]
    print(scores.shape)

    scores = scores.tolist()
    ordered_scores = scores.copy()
    ordered_scores.sort()
    ordered_scores.reverse()
    #print(ordered_scores)
    print(len(scores))
    for i in range(0,len(scores)):
        #print(scores.index(ordered_scores[i]))
        shutil.copyfile(f'mut/{scores.index(ordered_scores[i])}.txt', f'mutFinal/{i}.txt', )
    del scores
    return not_killed_count