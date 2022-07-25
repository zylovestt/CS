import numpy as np
import math
import CS_ENV
import AC
import torch
import rl_utils
from TEST import model_test

np.random.seed(1)
torch.manual_seed(0)
lr = 1*1e-4
num_episodes = 200
gamma = 0.99
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
seed=[i for i in range(10) for _ in range(20)]
'''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
np.set_printoptions(2)
pro_dic={}
pro_dic['F']=(0.9,0.99)
pro_dic['Q']=(0.7,1)
pro_dic['er']=(10,20)
pro_dic['econs']=(1,5)
pro_dic['rcons']=(1,5)
pro_dic['B']=(10,20)
pro_dic['p']=(10,20)
pro_dic['g']=(10,20)
def fx():
    h=np.random.random()
    def g(x):
        t=100*h*math.sin(h*x/10)+10
        return t
    return g
def fy():
    h=np.random.random()
    def g(x):
        t=50*h*math.sin(h*x/5)-10
        return t
    return g
pro_dic['x']=fx
pro_dic['y']=fy
pro_dic['w']=1
pro_dic['alpha']=2
pro_dic['twe']=(0,0)
pro_dic['ler']=(0,0)
num_pros=10
pro_dics=[CS_ENV.fpro_config(pro_dic) for _ in range(num_pros)]
task_dic={}
task_dic['ez']=(10,20)
task_dic['rz']=(10,20)
maxnum_tasks=4
task_dics=[CS_ENV.ftask_config(task_dic) for _ in range(maxnum_tasks)]
job_d={}
job_d['time']=(1,9)
job_d['womiga']=(0.5,1)
job_d['sigma']=(0.5,1)
job_d['num']=(1,maxnum_tasks)
job_dic=CS_ENV.fjob_config(job_d)
loc_config=CS_ENV.floc_config()
z=['Q','T','C','F']
lams={}
lams['T']=1*1e-2
lams['Q']=-1*1e-2
lams['F']=-1*1e-2
lams['C']=1*1e-2
bases={x:1 for x in z}
bases['T']=15
bases['Q']=-1
bases['C']=10
env=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
    job_dic,loc_config,lams,100,bases,seed)
#env.set_random_const_()
state=env.reset()

w=(state[0].shape,state[1].shape)
agent=AC.ActorCritic_Double_softmax(w,maxnum_tasks,lr,1,gamma,device,
    clip_grad=1e-2,beta=1e-2,n_steps=4,mode='gce',labda=0.95,eps=1e-5)
rl_utils.train_on_policy_agent(env,agent,num_episodes,10,10)
torch.save(agent.agent.state_dict(), "./data/CS_AC_model_parameter.pkl")
agent.writer.close()
#print(agent.cri_loss)
l1=model_test(env,agent,5,1)
print('next_agent##################################################')
r_agent=CS_ENV.RANDOM_AGENT(maxnum_tasks)
l2=model_test(env,r_agent,5,1)
print(l1,l2)
'''40.25833197208368 48.83124857335235'''