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
num_episodes = 20
gamma = 0.99
num_pros=10
maxnum_tasks=4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
tseed=[np.random.randint(0,5) for _ in range(1000)]
seed=[np.random.randint(0,5) for _ in range(20)]
'''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
np.set_printoptions(2)
pro_dic={}
pro_dic['F']=(0.9,0.99)
pro_dic['Q']=(0.9,1)
pro_dic['er']=(0.5,1)
pro_dic['econs']=(0.5,1)
pro_dic['rcons']=(0.5,1)
pro_dic['B']=(0.5,1)
pro_dic['p']=(0.5,1)
pro_dic['g']=(0.5,1)
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
pro_dics=[CS_ENV.fpro_config(pro_dic) for _ in range(num_pros)]
task_dic={}
task_dic['ez']=(0.5,1)
task_dic['rz']=(0.5,1)
task_dics=[CS_ENV.ftask_config(task_dic) for _ in range(maxnum_tasks)]
job_d={}
job_d['time']=(1,9)
job_d['womiga']=(0.5,1)
job_d['sigma']=(0.5,1)
job_d['num']=(1,maxnum_tasks)
job_dic=CS_ENV.fjob_config(job_d)
loc_config=CS_ENV.floc_config()
'''z=['Q','T','C','F']
lams={}
lams['T']=1*1e-2
lams['Q']=-1*1e-2
lams['F']=-1*1e-2
lams['C']=1*1e-2
bases={x:1 for x in z}
bases['T']=15
bases['Q']=1
bases['C']=10
env=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
    job_dic,loc_config,lams,100,bases,seed,seed)
state=env.reset()
w=(state[0].shape,state[1].shape)'''
z=['Q','T','C','F']
lams={}
lams['T']=1*1e-1
lams['Q']=-1*1e-1
lams['F']=-1*1e-1
lams['C']=1*1e-1
bases={x:1 for x in z}
env_c=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,100,bases,bases,seed,tseed)
state=env_c.reset()
W=(state[0].shape,state[1].shape)
r_agent=CS_ENV.RANDOM_AGENT(maxnum_tasks)
model_test(env_c,r_agent,1)
for key in env_c.bases:
    env_c.tar_dic[key].sort()
    g=np.array(env_c.tar_dic[key],dtype='float32')
    l=len(g)
    env_c.bases[key]=g[l//2]
    env_c.bases_fm[key]=g[l*3//4]-g[l//4]+1
for key in env_c.bases:
    env_c.tar_dic[key]=[]
    env_c.tarb_dic[key+'b']=[]
bases_fm=env_c.bases_fm
model_test(env_c,r_agent,1)

agent=AC.ActorCritic_Double_softmax(W,maxnum_tasks,lr,1,gamma,device,
    clip_grad='max',beta=1e-2,n_steps=4,mode='gce',labda=0.95,eps=1e-5)
rl_utils.train_on_policy_agent(env_c,agent,num_episodes,10,10)
torch.save(agent.agent.state_dict(), "./data/CS_AC_model_parameter.pkl")
agent.writer.close()
#print(agent.cri_loss)
l1=model_test(env_c,agent,10)
print('next_agent'+'#'*60)
#r_agent=CS_ENV.RANDOM_AGENT(maxnum_tasks)
l2=model_test(env_c,r_agent,10)
print(l1,l2)
'''40.25833197208368 48.83124857335235'''
