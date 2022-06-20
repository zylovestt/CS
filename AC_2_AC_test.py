import numpy as np
import ENV_AGENT
import AC
import torch
import rl_utils
from PRINT import Logger
from collections import deque
from TEST import model_test
from RANDOMAGENT import RANDOMAGENT_onehot
from copy import deepcopy

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=10
num_units=1
bs_cover=100
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(3000,4000),
        'sr':random_uniform_int(20,40),
        'tp':random_uniform_float(1,3),
        'pfr':random_uniform_int(50000,100000),
        'pf':random_uniform_int(200,5000),
        #'pf':F_pf,
        'plr':unit_loc(0,bs_cover),
        'pl':random_loc(0,bs_cover//10),
        'pd':random_uniform_float(2,6),
        'ps':random_uniform_float(3,5),
        'pbr':random_uniform_int(2000,5000),
        'pb':random_uniform_int(2000,4000),
        'ppr':random_uniform_int(200,300),
        'pp':random_uniform_int(100,200),
        'pg':random_uniform_int(20,40),
        'pcr':random_uniform_float(50,300),
        'pc':random_uniform_float(10,100),
        'whitenoise':1,
        'alpha':2}

config_in=deepcopy(config)
config_in['tp']=random_uniform_float(8,10)
np.random.seed(1)
torch.manual_seed(0)
lr = 1e-1
num_episodes = 1
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
num_subtasks=3
time_base=20
weights=np.ones(8)
weights[:]=1e-2
weights[0]/=1
weights[1]=0

deque_in_1=deque()
proin_1=ENV_AGENT.PROIN(config_in,1000,deque_in_1)
proin_1.run()
deque_out_1=deque()
env_agent_1=ENV_AGENT.ENV_AGENT(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
w=(env_agent_1.num_processors,env_agent_1.num_processor_attributes-1+env_agent_1.num_subtasks)
agent_1=AC.ActorCritic_Double(w,num_subtasks,lr,1,gamma,device,clip_grad=1,beta=0,n_steps=4,mode='gce',labda=0.95)
env_agent_1.set_agent(agent_1)
bigenv_1=ENV_AGENT.BIGENV_ONE([deque_in_1],[deque_out_1],env_agent_1)

deque_in_2=deque()
proin_2=ENV_AGENT.PROIN(config_in,1000,deque_in_2)
proin_2.run()
deque_out_2=deque()
env_agent_2=ENV_AGENT.ENV_AGENT(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
w=(env_agent_2.num_processors,env_agent_2.num_processor_attributes-1+env_agent_2.num_subtasks)
agent_2=AC.ActorCritic_Double(w,num_subtasks,lr,1,gamma,device,clip_grad=1,beta=0,n_steps=4,mode='gce',labda=0.95)
env_agent_2.set_agent(agent_2)
bigenv_2=ENV_AGENT.BIGENV_ONE([deque_in_2],[deque_out_2],env_agent_2)

deque_out_cat=deque()
env_agent_cat=ENV_AGENT.ENV_AGENT(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
w=(env_agent_cat.num_processors,env_agent_cat.num_processor_attributes-1+env_agent_cat.num_subtasks)
agent_cat=AC.ActorCritic_Double(w,num_subtasks,lr,1,gamma,device,clip_grad=1,beta=0,n_steps=4,mode='gce',labda=0.95)
env_agent_cat.set_agent(agent_cat)
bigenv_cat=ENV_AGENT.BIGENV_ONE([deque_out_1,deque_out_2],[deque_out_cat],env_agent_cat)


logger = Logger('AC2_'+str(agent_1.mode)+'_'+str(lr)+'.log')
bigenv_1.run()
bigenv_2.run()
rl_utils.train_on_policy_agent(bigenv_cat,bigenv_cat.agent,num_episodes,10)
agent_1.writer.close()
agent_2.writer.close()
agent_cat.writer.close()
