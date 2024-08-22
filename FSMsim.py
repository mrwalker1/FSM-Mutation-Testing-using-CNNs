def read_file_and_create_array(file_path):
    """
    :param file_path: This is the FSM file path
    :return: The FSM as array
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    fsm = {}
    for line in lines:
        elements = line.strip().split(' ')
        if len(elements) != 4:
            raise ValueError('Each line should have 4 characters separated by a space')

        startstate, input1, endstate, output = elements[0], elements[2], elements[1], elements[3]
        
        if startstate not in fsm:
            fsm[startstate] = {}

        if endstate not in fsm[startstate]:
            fsm[startstate][endstate] = []

        fsm[startstate][endstate].append(str(input1))
        fsm[startstate][endstate].append(str(output))

    return fsm

def sim(args):
    """
    This simulates the FSM with the inputs sequence.
    :param args: This contains all the arguments (file_path: Path of the FSM, sequ: The input sequence)
    :return: The output sequence the FSM produces
    """
    file_path, sequ = args
    fsm = read_file_and_create_array(file_path)

    
    current_state = "1"
    count = 0
    out_str = ""

    while count < len(sequ):
        found = False
        try:  
            for inp in fsm[current_state]:
                    #print(f'  {current_state}: {inp}: {fsm[current_state][inp][0]}: {fsm[current_state][inp][1]}')
                    
                    if sequ[count] in fsm[current_state][inp]:
                        #print(fsm[current_state][second_element][0][1])
                                            
                        pos = 1
                        if len(fsm[current_state][inp])>2:
                            tmp = round(len(fsm[current_state][inp])/2)
                            i=0
                            while i <=tmp*2:
                                if fsm[current_state][inp][i]== sequ[count]:
                                    pos = i+1
                                    break
                                i+=2
                        out_str += fsm[current_state][inp][pos]+","
                        current_state = str(inp)
                        count+=1

                        found = True
                        break
            if not found:
                out_str+="NaN"
                return out_str
        except:
            print(fsm)
            
            print(file_path)

    return out_str
if __name__ == "__main__":
    file_path = 'SmallestFSM.txt'
    sequ = str(input("Enter input sequence: "))
    sim([file_path, sequ])
