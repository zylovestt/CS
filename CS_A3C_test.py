import numpy as np
import math
import CS_ENV
import AC
import torch
import rl_utils
from PRINT import Logger
from TEST import model_test
import torch.multiprocessing as mp
import AGENT_NET
import os

if __name__=='__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    np.random.seed(1)
    torch.manual_seed(0)
    lr = 1*1e-4
    num_episodes = 10
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")

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
    num_pros=100
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
    env=CS_ENV.CSENV(pro_dics,maxnum_tasks,task_dics,
        job_dic,loc_config,lams,100,bases)
    env.set_random_const_()
    state=env.reset()

    w=(state[0].shape,state[1].shape)
    train_queue=mp.Queue(4)
    net=AGENT_NET.DoubleNet_softmax(w,maxnum_tasks).to(device)
    net.share_memory()
    optimizer=torch.optim.NAdam(net.parameters(),lr=lr,eps=1e-8)
    CUDA_LAUNCH_BLOCKING=1
    agent=AC.A3C_worker(w,maxnum_tasks,lr,1,gamma,device,
        clip_grad='max',beta=0,n_steps=4,mode='gce',labda=0.95,proc_name='f1',
        train_queue=train_queue,env=env,num_episodes=num_episodes,max_steps=10,net=net)
    logger = Logger('AC_'+str(agent.mode)+'_'+str(lr)+'.log')
    #agent.upload()
    data_proc_list = []
    for proc_idx in range(1):
        data_proc = mp.Process(target=agent.upload)
        data_proc.run()
        data_proc_list.append(data_proc)

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

            if step_idx % 5 == 0:
                for param, grad in zip(net.parameters(),
                                        grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

    logger.reset()
