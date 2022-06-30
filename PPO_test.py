import numpy as np
import ENV
import torch
import rl_utils
from matplotlib import pyplot as plt
import PPO

np.random.seed(1)
torch.manual_seed(0)
lr = 1*1e-4
num_episodes = 100
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=10
num_units=5
bs_cover=2000
F_pf=lambda x:np.array([10,10000])
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(3000,4000),
        'sr':random_uniform_int(20,40),
        'tp':random_uniform_float(1,2),
        'pfr':random_uniform_int(50000,100000),
        'pf':random_uniform_int(200,5000),
        #'pf':F_pf,
        'plr':unit_loc(0,bs_cover),
        'pl':random_loc(0,bs_cover//100),
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

num_subtasks=4
time_base=20
weights=np.ones(8)
weights[:]=1e-2
weights[0]/=1
weights[1]=0
env=ENV.ENVONE(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
env.set_random_const_()
env.cdma=True
lmbda = 0.95
epochs = 3
eps = 0.2

w=(env.num_processors,env.num_processor_attributes-1+env.num_subtasks)
agent = PPO.PPO(w,num_subtasks, lr,1,  gamma, device,'max',lmbda,epochs, eps)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes,10)

'''episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on my_env')
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on my_env')
plt.show()'''