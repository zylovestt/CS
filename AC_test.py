import numpy as np
import ENV
import AC
import torch
import rl_utils
from PRINT import Logger
from TEST import model_test
from RANDOMAGENT import RANDOMAGENT_onehot

np.random.seed(1)
torch.manual_seed(0)
lr = 1*1e-4
num_episodes = 100
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

random_uniform_int=lambda low,high:(lambda x:np.random.randint(low,high,x))
random_uniform_float=lambda low,high:(lambda x:np.random.uniform(low,high,x))
random_loc=lambda low,high:(lambda x:np.random.choice(np.arange(low,high),x,replace=False).astype('float'))
unit_loc=lambda s,e:(lambda x:np.linspace(s,e,x+1)[:-1])
num_cars=20
num_units=5
bs_cover=2000
F_pf=lambda x:np.array([10,10000])
config={'source':random_uniform_int(num_units,num_cars+num_units),
        'sc':random_uniform_int(3000,4000),
        'sr':random_uniform_int(20,40),
        'tp':random_uniform_float(1,3),
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

num_subtasks=10
time_base=20
weights=np.ones(8)
weights[:]=1e-2
weights[0]/=1
weights[1]=0
env=ENV.ENVONE(time_base,weights,num_processors=num_cars+num_units,
num_subtasks=num_subtasks,num_roadsideunits=num_units,basestation_cover=bs_cover,config=config)
env.set_random_const_()
env.cdma=True
state=env.reset()
w=(state[0].shape,state[1].shape)
#w=(env.num_processors,env.num_processor_attributes-1+env.num_subtasks)
#agent=AC.ActorCritic(w,num_subtasks,actor_lr,critic_lr,gamma,device,clip_grad=1,beta=0,conv=1)
agent=AC.ActorCritic_Double(w,num_subtasks,lr,1,gamma,device,clip_grad='max',beta=0,n_steps=4,mode='gce',labda=0.95)
#agent.agent.load_state_dict(torch.load("./data/model_parameter.pkl"))
logger = Logger('AC_'+str(agent.mode)+'_'+str(lr)+'.log')
return_list=rl_utils.train_on_policy_agent(env,agent,num_episodes,10)
torch.save(agent.agent.state_dict(), "./data/model_parameter.pkl")
agent.writer.close()

l1=model_test(env,agent,5)
print('next_agent##################################################')
r_agent=RANDOMAGENT_onehot(w,num_subtasks,num_units)
l2=model_test(env,r_agent,5)
print(l1,l2)
logger.reset()
'''plt.plot(agent.agent_loss)
plt.savefig('agent_loss')
plt.show()
plt.plot(agent.cri_loss)
plt.savefig('cri_loss')
plt.show()
plt.plot(agent.act_loss)
plt.savefig('act_loss')
plt.show()
plt.plot(agent.eposub_loss)
plt.savefig('eposub_loss')
plt.show()
plt.plot(agent.epopri_loss)
plt.savefig('epopri_loss')
plt.show()
plt.plot(agent.ac_loss)
plt.savefig('ac_loss')
plt.show()
print(np.array(l2).sum())

epoisodes_list=list(range(len(return_list)))
plt.plot(epoisodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on MY_ENV')
plt.show()

mv_return=rl_utils.moving_average(return_list,9)
plt.plot(epoisodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on MY_ENV')
plt.show()'''
