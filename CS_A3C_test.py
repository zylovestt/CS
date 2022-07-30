import numpy as np
import math
import CS_ENV
import AC
import torch
from TEST import model_test
import torch.multiprocessing as mp
import AGENT_NET
import time
import os

np.random.seed(1)
torch.manual_seed(0)
LR=1e-4
NUM_EPISODES=20
ENV_STEPS=100
MAX_STEPS=10
NUM_PROCS=4
NUM_ENVS=1
QUEUE_SIZE=NUM_PROCS
TRAIN_BATCH=1
NUM_PROCESSORS=10
MAXNUM_TASKS=4
BATCH_SIZE=1
GAMMA=0.99
EPS=1e-8
CYCLSES=10
DEVICE="cpu"
TANH=True
#SEED=[i for i in range(5) for _ in range(10)]
#SEED.extend([i for _ in range(10) for i in range(5)])
#SEED=[i for i in range(5) for _ in range(10)]
TSEED=[np.random.randint(0,1000) for _ in range(1000)]
SEEDS=[[[np.random.randint(0,1000) for _ in range(1000)] for _ in range(NUM_ENVS)] for _ in range(NUM_PROCS)]
#SEED=[1 for _ in range(1000)]
#print('tseed:',TSEED)
#print('seed:',SEEDS)
np.set_printoptions(2)
pro_dic={}
'''pro_dic['F']=(0.9,0.99)
pro_dic['Q']=(0.7,1)
pro_dic['er']=(1,200)
pro_dic['econs']=(1,10)
pro_dic['rcons']=(1,10)
pro_dic['B']=(1,200)
pro_dic['p']=(1,200)
pro_dic['g']=(1,200)'''
pro_dic['F']=(0.9,0.99)
pro_dic['Q']=(0.9,1)
pro_dic['er']=(19,20)
pro_dic['econs']=(4,5)
pro_dic['rcons']=(4,5)
pro_dic['B']=(19,20)
pro_dic['p']=(19,20)
pro_dic['g']=(19,20)
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
pro_dics=[CS_ENV.fpro_config(pro_dic) for _ in range(NUM_PROCESSORS)]
task_dic={}
'''task_dic['ez']=(1,200)
task_dic['rz']=(1,200)'''
task_dic['ez']=(1,20)
task_dic['rz']=(1,20)
task_dics=[CS_ENV.ftask_config(task_dic) for _ in range(MAXNUM_TASKS)]
job_d={}
job_d['time']=(1,9)
job_d['womiga']=(0.5,1)
job_d['sigma']=(0.5,1)
job_d['num']=(1,MAXNUM_TASKS)
job_dic=CS_ENV.fjob_config(job_d)
loc_config=CS_ENV.floc_config()
z=['Q','T','C','F']
lams={}
lams['T']=1*1e-1
lams['Q']=-1*1e-1
lams['F']=-1*1e-1
lams['C']=1*1e-1
bases={x:1 for x in z}
env_c=CS_ENV.CSENV(pro_dics,MAXNUM_TASKS,task_dics,
        job_dic,loc_config,lams,ENV_STEPS,bases,bases,SEEDS[0],TSEED)
state=env_c.reset()
W=(state[0].shape,state[1].shape)
r_agent=CS_ENV.RANDOM_AGENT(MAXNUM_TASKS)
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

def data_func(proc_name,net,train_queue,id):
    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    frame_idx=0
    ts_time=time.time()

    f_env=lambda x:CS_ENV.CSENV(pro_dics,MAXNUM_TASKS,task_dics,
        job_dic,loc_config,lams,ENV_STEPS,bases,bases_fm,SEEDS[id][x],TSEED)

    '''input_shape,num_subtasks,weights,gamma,device,clip_grad,beta,n_steps,mode,labda,proc_name'''
    worker=AC.ActorCritic_Double_softmax_worker(W,MAXNUM_TASKS,1,GAMMA,DEVICE,
        clip_grad=1e-2,beta=1e-2,n_steps=0,mode='gce',labda=0.95,proc_name=proc_name)

    worker.agent=net
    envs=[f_env(i) for i in range(NUM_ENVS)]
    for i,env in enumerate(envs):
        env.name=proc_name+' '+str(i)
        #env.set_random_const_()

    worker.set_nolocal_update()

    done=False
    state=[env.reset() for env in envs]
    episode_return=[0 for _ in range(NUM_ENVS)]
    return_list=[]
    i_episode=0
    grads_l=[]
    while i_episode<NUM_EPISODES:
        for i,env in enumerate(envs):
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'overs': []}
            step=0
            while not done and step<MAX_STEPS:
                step+=1
                frame_idx+=1
                action = worker.take_action(state[i])
                next_state, reward, done, over, _ = env.step(action)
                transition_dict['states'].append(state[i])
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['overs'].append(over)
                state[i] = next_state
                episode_return[i] += reward
            if done:
                return_list.append(episode_return[i])
                worker.writer.add_scalar(tag='return',scalar_value=episode_return[i],global_step=i_episode)
                i_episode+=1
                if i_episode%CYCLSES==0:
                    s=frame_idx/(time.time()-ts_time)
                    print('{}: speed:{}'.format(proc_name,s))
                    worker.writer.add_scalar(tag='speed',scalar_value=s,global_step=i_episode)
                    frame_idx,ts_time=0,time.time()
                    test_reward=model_test(env,worker,1)
                    print('{}: episode:{} test_reward:{}'.format(proc_name,i_episode,test_reward))
                    worker.writer.add_scalar('test_reward',test_reward,i_episode)
                    print('{}: episode:{} reward:{}'.format(proc_name,i_episode,np.mean(return_list[-CYCLSES:])))
                state[i] = env.reset()
                done = False
                episode_return[i] = 0
            grads_l.append(worker.update(transition_dict))
        if i_episode%BATCH_SIZE==0:
            for k in range(1,len(grads_l)):
                for grad0,gradk in zip(grads_l[0],grads_l[k]):
                    grad0+=gradk
            train_queue.put(grads_l[0])
            grads_l.clear()
    worker.writer.close()
    train_queue.put(None)

if __name__=='__main__':
    mp.set_start_method('spawn',force=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    DEVICE = "cpu"

    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    train_queue=mp.Queue(QUEUE_SIZE)
    #net=AGENT_NET.DoubleNet_softmax(W,MAXNUM_TASKS).to(DEVICE)
    net=AGENT_NET.DoubleNet_softmax_simple(W,MAXNUM_TASKS,TANH).to(DEVICE)
    net.share_memory()
    optimizer=torch.optim.NAdam(net.parameters(),lr=LR,eps=EPS)
    
    data_proc_list = []
    for proc_idx in range(NUM_PROCS):
        args=(str(proc_idx), net, train_queue,proc_idx)
        p = mp.Process(target=data_func,args=args)
        p.start()
        data_proc_list.append(p)

    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer,
                                            train_entry):
                    tgt_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(),
                                        grad_buffer):
                    param.grad = torch.FloatTensor(grad/(TRAIN_BATCH*NUM_ENVS*BATCH_SIZE)).to(DEVICE)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

    f_worker=AC.ActorCritic_Double_softmax_worker(W,MAXNUM_TASKS,1,GAMMA,DEVICE,
            clip_grad=1e-2,beta=0,n_steps=0,mode='gce',labda=0.95,proc_name='finally')
    f_worker.agent=net
    l1=model_test(env_c,f_worker,10)
    print('next_agent'+'#'*60)
    #r_agent=CS_ENV.RANDOM_AGENT(MAXNUM_TASKS)
    l2=model_test(env_c,r_agent,10)
    print(l1,l2)
    torch.save(net.state_dict(), "./data/CS_A3C_model_parameter.pkl")
'''30.085082623437224 48.83124857335235'''