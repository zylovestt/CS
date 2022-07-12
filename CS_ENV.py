import numpy as np

class PROCESSOR:
    def __init__(self,config:dict):
        '''F,Q,er,econs,rcons,B,p,g,d,w,alpha,twe,ler,twr'''
        self.pro_dic={}
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
    
    def cal_PF(self):
        self.PF=(self.Exe+1)/(self.Exe+self.UExe+2)
    
    def cal_Aq(self):
        self.Aq=self.sum_Aq/(self.Nk+1)
    
    def __call__(self,tin:float,task:dict,sigma:float):
        te=task['ez']/self.pro_dic['er']
        twe=self.pro_dic['twe']
        ler=self.pro_dic['ler']
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
            r=self.pro_dic['B']*np.log2(
                1+self.pro_dic['p']*self.pro_dic['h']/
                (self.pro_dic['d'](t)**self.pro_dic['alpha']
                *self.pro_dic['w']**2))
            tr=task['rz'][i]/r
            ler=twr+tr
        self.cal_PF()
        Q*=sigma
        self.sum_Aq+=Q
        self.cal_Aq()
        return Q,twe+ler,np.sum(te)*self.pro_dic['econs']+np.sum(tr)*self.pro_dic['rcons'],finish

class PROCESSORS:
    def __init__(self,pro_configs:list):
        self.num_pros=len(pro_configs)
        self.pros=[]
        for pro_config in pro_configs:
            self.pros.append(PROCESSOR(pro_config))
    
    def __call__(self,tin:float,tasks:dict,action:np.ndarray,sigma:float,womiga:float):
        for i,rz in enumerate(tasks['rz']):
            if not rz:
                tasks_num=i+1
                break
        tasks['ez']=tasks['ez'][:tasks_num]
        tasks['rz']=tasks['rz'][:tasks_num]
        act_list=[]
        for i in range(tasks_num):
            act_list.append(i,action[0][i],action[1][i])
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
        task_time*=womiga
        return Q,task_time,cons,finish

class JOB:
    def __init__(self,maxnum_tasks:int):
        self.tasks={}
        
    def __call__(self,task_configs,job_config):
        num=job_config['num']()
        for i,config in enumerate(task_configs):
            if i<num:
                for k in config:
                    self.tasks[k].append(config[k]())
            else:
                for k in self.tasks:
                    self.tasks[k].append(0)
        

