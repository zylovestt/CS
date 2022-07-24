import numpy as np
import math
import CS_ENV
import AC
import torch
import rl_utils
from PRINT import Logger
from TEST import model_test
import torch.multiprocessing as mp
import multiprocessing as mu
from copy import deepcopy as dp
import AGENT_NET
import os

np.random.seed(1)
torch.manual_seed(0)
np.set_printoptions(2)
lr = 1*1e-4
num_episodes = 100
max_steps=10
num_procs=2
queue_size=100
train_batch=1
gamma = 0.98
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
lams={x:1 for x in z}
lams['Q']=-1
lams['F']=-1
lams['C']=1
bases={x:1 for x in z}
bases['T']=15
bases['Q']=-1
bases['C']=10

def data_func(net, device, train_queue):
    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    env=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,100,bases)
    env.set_random_const_()
    state=env.reset()

    w=(state[0].shape,state[1].shape)

    worker=AC.ActorCritic_Double_softmax33(w,maxnum_tasks,lr,1,gamma,device,
        clip_grad='max',beta=0,n_steps=4,mode='gce',labda=0.95)

    worker.agent=net
    worker.env=env
    worker.num_episodes=num_episodes
    worker.max_steps=max_steps
    worker.set_nolocal_update()

    env=worker.env
    return_list=[]
    done=False
    state=env.reset()
    episode_return=0
    i_episode=0
    while i_episode<worker.num_episodes:
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'overs': []}
        step=0
        while not done and step<worker.max_steps:
            step+=1
            action = worker.take_action(state)
            next_state, reward, done, over, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['overs'].append(over)
            state = next_state
            episode_return += reward
        if done:
            state = env.reset()
            done = False
            return_list.append(episode_return)
            #writer.add_scalar(tag='return',scalar_value=episode_return,global_step=i_episode)
            episode_return = 0
            i_episode+=1
            if (i_episode % 10 == 0):
                test_reward=model_test(env,worker,1,1)
                print('episode:{}, test_reward:{}'.format(i_episode,test_reward))
                #writer.add_scalar('test_reward',test_reward,i_episode)
                print('episode:{}, reward:{}'.format(i_episode,np.mean(return_list[-10:])))
        grads=worker.update(transition_dict)
        train_queue.put(grads)

if __name__=='__main__':
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    lr = 1*1e-4
    num_episodes = 4
    gamma = 0.98
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
    env=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,100,bases)
    env.set_random_const_()
    state=env.reset()

    w=(state[0].shape,state[1].shape)
    
    train_queue=mp.Queue(queue_size)
    net=AGENT_NET.DoubleNet_softmax(w,maxnum_tasks).to(device)
    net.share_memory()
    optimizer=torch.optim.NAdam(net.parameters(),lr=lr,eps=1e-8)
    
    data_proc_list = []
    args=(net, device, train_queue)
    for proc_idx in range(num_procs):
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

            if step_idx % train_batch == 0:
                for param, grad in zip(net.parameters(),
                                        grad_buffer):
                    param.grad = torch.FloatTensor(grad/train_batch).to(device)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

