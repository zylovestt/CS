import numpy as np
from collections import OrderedDict

rui=lambda u:(lambda:float(np.random.randint(u[0],u[1])))
ruf=lambda u:(lambda:float(np.random.uniform(u[0],u[1])))

def pro_config(dic):
    config={}
    i=['er','econs','rcons','B','p','g']
    f=['F','Q','twe','ler']
    for item in i:
        config[item]=rui(dic[item])
    for item in f:
        config[item]=ruf(dic[item])
    config['w']=float(dic['w'])
    config['alpha']=float(dic['alpha'])
    config['d']=dic['d']
    return config

def task_config(dic):
    config={}
    i=['rz','ez']
    for item in i:
        config[item]=rui(dic[item])
    return config

def job_config(dic):
    config={}
    f=['time','womiga','sigma']
    for item in f:
        config[item]=ruf(dic[item])
    return config

def loc_config():
    def generate(num_pros,maxnum_tasks):
        task_num=np.random.randint(1,maxnum_tasks)
        num_pro_choices=np.random.randint(1,num_pros,task_num)
        loc=np.zeros((num_pros,maxnum_tasks),'int')
        for i in range(task_num):
            num_pro_choice=num_pro_choices[i]
            pro_choice=np.random.choice(np.arange(1,num_pros,dtype='int'),num_pro_choice,False)
            loc[pro_choice,i]=1
        return loc
    return generate

class PROCESSOR:
    def __init__(self,config:dict):
        '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler'''
        self.pro_dic=OrderedDict()
        for k in config:
            if callable(config) and not (k=='d' or k=='Q'):
                self.pro_dic[k]=config[k]()
            else:
                self.pro_dic[k]=config[k]
        self.Exe=0
        self.UExe=0
        self.cal_PF()
        self.sum_Aq=0
        self.Nk=0
        self.cal_Aq()
        self.t=0
    
    def cal_PF(self):
        self.PF=(self.Exe+1)/(self.Exe+self.UExe+2)
    
    def cal_Aq(self):
        self.Aq=self.sum_Aq/(self.Nk+1)
    
    def __call__(self,tin:float,task:dict,sigma:float):
        te=task['ez']/self.pro_dic['er']
        tp=tin-self.t
        self.t=tin
        twe=self.pro_dic['twe']
        ler=self.pro_dic['ler']
        ler=min(max(ler+twe-tp,0),ler)
        twe=max(twe-tp,0)
        Q,finish=0,True
        for i in range(len(te)):
            if np.random.rand()>self.pro_dic['F']:
                self.UExe+=1
                finish=False
            else:
                Q+=self.pro_dic['Q']()
                self.Nk+=1
                self.Exe+=1
            twe+=te[i]
            twr=max(ler-te[i],0)
            t=tin+twe+twr
            tr=self.cal_tr(task['rz'][i],t)
            ler=twr+tr
        self.pro_dic['twe']=twe
        self.pro_dic['ler']=ler
        self.cal_PF()
        Q*=sigma
        self.sum_Aq+=Q
        self.cal_Aq()
        return Q,twe+ler,np.sum(te)*self.pro_dic['econs']+np.sum(tr)*self.pro_dic['rcons'],finish
    
    def cal_tr(self,rz,t):
        r=self.pro_dic['B']*np.log2(
                1+self.pro_dic['p']*self.pro_dic['h']/
                (self.pro_dic['d'](t)**self.pro_dic['alpha']
                *self.pro_dic['w']**2))
        return rz/r

class PROCESSORS:
    def __init__(self,pro_configs:list):
        self.num_pros=len(pro_configs)
        self.pros=[PROCESSOR(pro_config) for pro_config in pro_configs]
    
    def __call__(self,tin:float,tasks:dict,action:np.ndarray,womiga:float,sigma:float):
        for i,rz in enumerate(tasks['rz']):
            if not rz:
                tasks_num=i+1
                break
        tasks['ez']=tasks['ez'][:tasks_num]
        tasks['rz']=tasks['rz'][:tasks_num]
        act_list=[(i,action[0][i],action[1][i]) for i in range(tasks_num)]
        act_list=sorted(act_list,key=lambda x:x[-1])
        Q,task_time,cons,finish=0,0,0,True
        for pro in self.pros:
            task={}
            task['ez'],task['rz']=[],[]
            for item in act_list:
                if item[1]==i:
                    task['ez'].append(tasks['ez'][item[0]])
                    task['rz'].append(tasks['rz'][item[0]])
            if len(task['ez']):
                Q1,task_time1,cons1,finish1=pro(tin,task,sigma)
                if not finish1:
                    finish=finish1
                Q+=Q1
                task_time=max(task_time,task_time1)
                cons+=cons1
        return Q,task_time*womiga,cons,finish

class JOB:
    def __init__(self,maxnum_tasks):
        self.maxnum_tasks=maxnum_tasks
        self.job_index=0
        self.tin=0

    def __call__(self,task_configs:list,job_config:dict):
        self.job_index+=1
        num=job_config['num']()
        tasks={}
        for i,config in enumerate(task_configs):
            if i<num:
                for k in config:
                    tasks[k].append(config[k]())
            else:
                for k in tasks:
                    tasks[k].append(0)
        self.tin+=job_config['time']()
        womiga=job_config['womiga']()
        sigma=job_config['sigma']()
        return self.tin,tasks,womiga,sigma

class JOBPPROS:
    def __init__(self,job:JOB,pros:PROCESSORS,lams:list):
        '''lams:Q,T,C,F'''
        self.job=job
        self.processor=pros
        self.lams=lams
        self.tar_dic=OrderedDict()
        self.tar_dic['Q']=[]
        self.tar_dic['T']=[]
        self.tar_dic['C']=[]
        self.tar_dic['F']=[]
        self.sum_tar=[]
    
    def send(self,task_configs,job_config,loc_config):
        self.tin,self.tasks,self.womiga,self.sigma=self.job(task_configs,job_config)
        task_loc=loc_config(self.processor.num_pros,self.job.maxnum_tasks)
        pro_status=[]
        for pro in self.processor.pros:
            items=[value for value in pro.pro_dic if not callable(value)]
            items.extend([pro.PF,pro.Aq,pro.pro_dic['d'](self.job.tin)-100])
            pro_status.append(items)
        pro_status=np.concatenate((np.array(pro_status),task_loc),1).reshape(1,1,self.processor.num_pros,-1)
        task_status=[]
        for item in self.tasks.values():
            task_status.extend(item)
        task_status.extend([self.womiga,self.sigma])
        task_status=np.array(task_status)
        return pro_status,task_status

    def accept(self,action):
        R=self.processor(self.tin,self.tasks,action,self.womiga,self.sigma)
        t=0
        for item,lam,r in zip(self.tar_dic.values(),self.lams,R):
            item.append(r)
            t+=lam*r
        self.sum_tar.append(t)
        
