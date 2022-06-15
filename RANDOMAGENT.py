import numpy as np

class RANDOMAGENT():
    def __init__(self,input_shape,num_subtasks):
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        
    def take_action(self,state):
        action=np.zeros((2,self.num_subtasks),dtype='int')
        sub_loc=state[0][0,0,:,-self.num_subtasks:]
        sub_loc=sub_loc[::-1]
        action[0]=-sub_loc.argmin(axis=0)+self.num_processors-1
        return action

class RANDOMAGENT_onehot():
    def __init__(self,input_shape,num_subtasks,num_units):
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.num_units=num_units
        
    def take_action(self,state):
        action=np.zeros((2,self.num_subtasks),dtype='int')
        sub_loc=state[0][0,0,:,-self.num_subtasks:]
        for i in range(self.num_subtasks):
            while True:
                pro=np.random.choice(range(self.num_processors))
                if sub_loc[pro][i]:
                    action[0][i]=pro
                    break
        return action
