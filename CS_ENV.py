import numpy as np
class PROCESSOR:
    def __init__(self,pro_config:dict):
        '''F,Q,er,rr,econs,rcons,B,p,g,d,w,alpha,twe,ler,twr'''
        self.pro_dic={}
        for k in pro_config:
            if callable(pro_config) and not (k=='d' or k=='Q'):
                self.pro_dic[k]=pro_config[k][1](1)
            else:
                self.pro_dic[k]=pro_config[k][1]
    
    def __call__(self,tin,task:dict):
        if np.random.rand()>self.pro_dic['F']:
            return (False)
        Q=self.pro_dic['Q'](1)
        te=task['ez']/self.pro_dic['er']
        tr=task['er']/self.pro_dic['rr']
        twe=self.pro_dic['twe']
        ler=self.pro_dic['ler']
        for i in range(len(te)):
            twe+=te[i]
            twr=max(ler-te[i],0)
            t=tin+twe+twr
            r=self.pro_dic['B']*np.log2(
                1+self.pro_dic['p']*self.pro_dic['h']/
                (self.pro_dic['d'](t)**self.pro_dic['alpha']
                *self.pro_dic['w']**2))
            ler=twr+tr[i]
        return Q,twe+ler,np.sum(te)*self.pro_dic['econs']+np.sum(tr)*self.pro_dic['rcons']



