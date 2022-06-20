import numpy as np
import ENV_AGENT
import ENV
import AC
import torch
import rl_utils
from PRINT import Logger
from matplotlib import pyplot as plt
from collections import deque
from TEST import model_test
from RANDOMAGENT import RANDOMAGENT_onehot
from copy import deepcopy

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=5
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
num_episodes = 100
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
num_subtasks=3
time_base=20
weights=np.ones(8)
weights[:]=1e-2
weights[0]/=1
weights[1]=0

deque_1=deque()
proin=ENV_AGENT.PROIN(config_in,1000,deque_1)
proin.run()

deque_2=deque()

env_agent=ENV_AGENT.ENV_AGENT(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)

w=(env_agent.num_processors,env_agent.num_processor_attributes-1+env_agent.num_subtasks)
#agent=AC.ActorCritic(w,num_subtasks,actor_lr,critic_lr,gamma,device,clip_grad=1,beta=0,conv=1)
agent=AC.ActorCritic_Double(w,num_subtasks,lr,1,gamma,device,clip_grad=1,beta=0,n_steps=4,mode='gce',labda=0.95)

env_agent.set_agent(agent)

bigenv=ENV_AGENT.BIGENV_ONE([deque_1],[deque_2],env_agent)

logger = Logger('AC2_'+str(agent.mode)+'_'+str(lr)+'.log')
return_list=rl_utils.train_on_policy_agent(bigenv,bigenv.agent,num_episodes,10)
#torch.save(agent.agent.state_dict(), "./data/model2_parameter.pkl")
agent.writer.close()

'''model_test(bigenv,bigenv.agent,1,num_subtasks,cycles=1)
print('next_agent##################################################')
r_agent=RANDOMAGENT(w,num_subtasks)
model_test(bigenv,r_agent,1,num_subtasks,cycles=1)'''

l1=model_test(bigenv,bigenv.agent,1,num_subtasks,cycles=1)
print('next_agent##################################################')
r_agent=RANDOMAGENT_onehot(w,num_subtasks,num_units)
l2=model_test(bigenv,r_agent,1,num_subtasks,cycles=1)
print(np.array(l1).sum(),np.array(l2).sum())

'''env=ENV.ENVONE(time_base,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
state=env.reset()
print('state:{}'.format(state[0]))
action=agent.take_action(state)
print('action:{}'.format(action))
next_state, reward, done, _ = env.step(action)
print('next_state:{}'.format(next_state[0]))'''