# FSM Mutation Testing using CNNs

## Running experiment 
```bash 
python3 "main.py" {FSM File Directory} {mode} {numberMutsPer} {End mut}
```
### Mode
mode 1 = Model 

mode 2 = Depth

mode 3 = Breath 

### End mut
True = The mutation will only happen to the last state. 

Flase = Mutation will happen to any state.

## Requirments
Python 3.10.12

keras~=2.15.0

numpy~=1.26.2

tensorflow~=2.15.0.post1

scikit-learn~=1.4.1.post1

enlighten~=1.12.4

matplotlib~=3.8.2

torch~=2.2.1

Pillow~=10.1.0

alive-progress~=3.1.5

imageio~=2.33.1
