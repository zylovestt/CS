import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class DoubleNet(nn.Module):
    def __init__(self,input_shape,num_subtasks,group_norm=False):
        super(DoubleNet,self).__init__()
        self.num_processors=input_shape[0][2]
        self.num_attributes=input_shape[0][-1]
        self.num_subtasks=num_subtasks
        self.group_norm=group_norm
        hs=192
        groups=32
        self.base_row=nn.Conv2d(1,hs,kernel_size=(1,input_shape[0][-1]),stride=1)
        self.base_col=nn.Conv2d(1,hs,kernel_size=(input_shape[0][2],1),stride=1)
        self.base_all=nn.Conv2d(1,3*hs,kernel_size=(input_shape[0][2],input_shape[0][-1]),stride=1)
        if group_norm:
            self.row_groupnorm=nn.GroupNorm(groups,hs)
            self.col_groupnorm=nn.GroupNorm(groups,hs)
            self.all_groupnorm=nn.GroupNorm(groups,3*hs)
        conv_out_size=self._get_conv_out(input_shape)+input_shape[1][1]
        self.fc=nn.Sequential(
            nn.PReLU(),
            nn.Linear(conv_out_size,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs))
        F=lambda x,y:nn.Sequential(
            nn.PReLU(),
            nn.Linear(x,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,y))
        self.critic_out=nn.Sequential(
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,1))
        self.policy_layer=nn.ModuleList([F(hs+self.num_processors,input_shape[0][2]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(hs,num_subtasks) for _ in range(num_subtasks)])
    
    def _get_conv_out(self,input_shape):
        s=input_shape[0][-2:]
        o_row=self.base_row(torch.zeros(1,1,*s))
        o_col=self.base_col(torch.zeros(1,1,*s))
        o_all=self.base_all(torch.zeros(1,1,*s))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape)+np.prod(o_all.shape))
    
    def forward(self,x):
        if self.group_norm:
            conv_out_row=self.row_groupnorm(self.base_row(x[0])).view(x[0].size()[0],-1)
            conv_out_col=self.col_groupnorm(self.base_col(x[0])).view(x[0].size()[0],-1)
            conv_out_all=self.all_groupnorm(self.base_all(x[0])).view(x[0].size()[0],-1)
        else:
            conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
            conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
            conv_out_all=self.base_all(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,conv_out_all,x[1]),1)
        out_fc=self.fc(conv_out)
        l1=[]
        for layer,i in zip(self.policy_layer,range(self.num_subtasks)):
            u=x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)
            v=torch.cat((out_fc,u),dim=1)
            w=layer(v)
            p=(1/(-u))+1+w
            z=F.softmax(p,dim=1)+1e-14
            if z.sum().isnan().item():
                print('net_here')
            l1.append(z)
        l2=[F.softmax(layer(out_fc),dim=1)+1e-14 for layer in self.prior_layer]
        critic=self.critic_out(out_fc)
        return (l1,l2),critic

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="prelu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class DoubleNet_softmax(nn.Module):
    def __init__(self,input_shape,num_subtasks,group_norm=False):
        super(DoubleNet_softmax,self).__init__()
        self.num_processors=input_shape[0][2]
        self.num_attributes=input_shape[0][-1]
        self.num_subtasks=num_subtasks
        self.group_norm=group_norm
        hs=192
        groups=32
        self.base_row=nn.Conv2d(1,hs,kernel_size=(1,input_shape[0][-1]),stride=1)
        self.base_col=nn.Conv2d(1,hs,kernel_size=(input_shape[0][2],1),stride=1)
        self.base_all=nn.Conv2d(1,3*hs,kernel_size=(input_shape[0][2],input_shape[0][-1]),stride=1)
        if group_norm:
            self.row_groupnorm=nn.GroupNorm(groups,hs)
            self.col_groupnorm=nn.GroupNorm(groups,hs)
            self.all_groupnorm=nn.GroupNorm(groups,3*hs)
        conv_out_size=self._get_conv_out(input_shape)+input_shape[1][1]
        self.fc=nn.Sequential(
            nn.PReLU(),
            nn.Linear(conv_out_size,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs))
        F=lambda x,y:nn.Sequential(
            nn.PReLU(),
            nn.Linear(x,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,y))
        self.critic_out=nn.Sequential(
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,1))
        self.policy_layer=nn.ModuleList([F(hs+self.num_processors,input_shape[0][2]) for _ in range(num_subtasks)])
        #self.policy_layer=nn.ModuleList([F(hs,input_shape[0][2]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(hs,num_subtasks) for _ in range(num_subtasks-1)])
    
    def _get_conv_out(self,input_shape):
        s=input_shape[0][-2:]
        o_row=self.base_row(torch.zeros(1,1,*s))
        o_col=self.base_col(torch.zeros(1,1,*s))
        o_all=self.base_all(torch.zeros(1,1,*s))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape)+np.prod(o_all.shape))
    
    def forward(self,x):
        if self.group_norm:
            conv_out_row=self.row_groupnorm(self.base_row(x[0])).view(x[0].size()[0],-1)
            conv_out_col=self.col_groupnorm(self.base_col(x[0])).view(x[0].size()[0],-1)
            conv_out_all=self.all_groupnorm(self.base_all(x[0])).view(x[0].size()[0],-1)
        else:
            conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
            conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
            conv_out_all=self.base_all(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,conv_out_all,x[1]),1)
        out_fc=self.fc(conv_out)
        l1=[]
        for layer,i in zip(self.policy_layer,range(self.num_subtasks)):
            u=x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)
            v=torch.cat((out_fc,u),dim=1)
            w=layer(v)
            #w=layer(out_fc)
            p=(1/(-u))+1+w
            if p.sum().isnan().item():
                print('net_here')
            l1.append(p)
        l2=[layer(out_fc) for layer in self.prior_layer]
        critic=self.critic_out(out_fc)
        return (l1,l2),critic

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="prelu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class DoubleNet_softmax_simple_fc(nn.Module):
    def __init__(self,input_shape,num_subtasks):
        super(DoubleNet_softmax_simple_fc,self).__init__()
        self.num_processors=input_shape[0][2]
        self.num_attributes=input_shape[0][-1]
        self.num_subtasks=num_subtasks
        hs=192
        #self.base_row=nn.Conv2d(1,hs,kernel_size=(1,input_shape[0][-1]),stride=1)
        #self.base_col=nn.Conv2d(1,hs,kernel_size=(input_shape[0][2],1),stride=1)
        self.base_all=nn.Conv2d(1,3*hs,kernel_size=(input_shape[0][2],input_shape[0][-1]),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+input_shape[1][1]
        self.fc=nn.Sequential(
            nn.PReLU(),
            nn.Linear(conv_out_size,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs))
        F=lambda x,y:nn.Sequential(
            nn.PReLU(),
            nn.Linear(x,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,y))
        self.critic_out=nn.Sequential(
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,1))
        #self.policy_layer=nn.ModuleList([F(hs+self.num_processors,input_shape[0][2]) for _ in range(num_subtasks)])
        self.policy_layer=nn.ModuleList([F(hs,input_shape[0][2]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(hs,num_subtasks) for _ in range(num_subtasks-1)])
    
    def _get_conv_out(self,input_shape):
        s=input_shape[0][-2:]
        #o_row=self.base_row(torch.zeros(1,1,*s))
        #o_col=self.base_col(torch.zeros(1,1,*s))
        o_all=self.base_all(torch.zeros(1,1,*s))
        return int(np.prod(o_all.shape))
        #return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        #conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
        #conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
        conv_out_all=self.base_all(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_all,x[1]),1)
        #conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        out_fc=self.fc(conv_out)
        l1=[]
        for layer,i in zip(self.policy_layer,range(self.num_subtasks)):
            u=x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)
            #v=torch.cat((out_fc,u),dim=1)
            #w=layer(v)
            w=layer(out_fc)
            p=(1/(-u))+1+w
            if p.sum().isnan().item():
                print('net_here')
            l1.append(p)
        l2=[layer(out_fc) for layer in self.prior_layer]
        critic=self.critic_out(out_fc)
        return (l1,l2),critic

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="prelu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class DoubleNet_softmax_simple0(nn.Module):
    def __init__(self,input_shape,num_subtasks,tanh=False):
        super(DoubleNet_softmax_simple0,self).__init__()
        self.num_processors=input_shape[0][2]
        self.num_attributes=input_shape[0][-1]
        self.num_subtasks=num_subtasks
        hs=256
        self.base_row=nn.Conv2d(1,hs,kernel_size=(1,input_shape[0][-1]),stride=1)
        self.base_col=nn.Conv2d(1,hs,kernel_size=(input_shape[0][2],1),stride=1)
        #self.base_all=nn.Conv2d(1,3*hs,kernel_size=(input_shape[0][2],input_shape[0][-1]),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+input_shape[1][1]
        self.tanh=tanh
        self.fc=nn.Sequential(
            nn.PReLU(),
            nn.Linear(conv_out_size,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs))
        if self.tanh:
            F=lambda x,y:nn.Sequential(
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(hs,y),
                nn.Tanh())
        else:
            F=lambda x,y:nn.Sequential(
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(hs,y))
        self.critic_out=nn.Sequential(
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,1))
        #self.policy_layer=nn.ModuleList([F(hs+self.num_processors,input_shape[0][2]) for _ in range(num_subtasks)])
        self.policy_layer=nn.ModuleList([F(hs,input_shape[0][2]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(hs,num_subtasks) for _ in range(num_subtasks-1)])
    
    def _get_conv_out(self,input_shape):
        s=input_shape[0][-2:]
        o_row=self.base_row(torch.zeros(1,1,*s))
        o_col=self.base_col(torch.zeros(1,1,*s))
        #o_all=self.base_all(torch.zeros(1,1,*s))
        #return int(np.prod(o_row.shape)+np.prod(o_col.shape)+np.prod(o_all.shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        conv_out_row=self.base_row(x[0]).view(x[0].size()[0],-1)
        conv_out_col=self.base_col(x[0]).view(x[0].size()[0],-1)
        #conv_out_all=self.base_all(x[0]).view(x[0].size()[0],-1)
        #conv_out=torch.cat((conv_out_row,conv_out_col,conv_out_all,x[1]),1)
        conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        out_fc=self.fc(conv_out)
        l1=[]
        for layer,i in zip(self.policy_layer,range(self.num_subtasks)):
            u=x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)
            #v=torch.cat((out_fc,u),dim=1)
            #w=layer(v)
            if self.tanh:
                w=3*layer(out_fc)
            else:
                w=layer(out_fc)
            p=(1/(-u))+1+w
            if p.sum().isnan().item():
                print('net_here')
            l1.append(p)
        l2=[3*layer(out_fc) if self.tanh else layer(out_fc)  for layer in self.prior_layer]
        critic=self.critic_out(out_fc)
        return (l1,l2),critic

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="prelu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class DoubleNet_softmax_simple(nn.Module):
    def __init__(self,input_shape,num_subtasks,tanh=False,depart=True):
        super(DoubleNet_softmax_simple,self).__init__()
        self.num_processors=input_shape[0][2]
        self.num_attributes=input_shape[0][-1]
        self.num_subtasks=num_subtasks
        self.depart=depart
        hs=256
        self.base_row_p=nn.Conv2d(1,hs,kernel_size=(1,input_shape[0][-1]),stride=1)
        self.base_col_p=nn.Conv2d(1,hs,kernel_size=(input_shape[0][2],1),stride=1)

        #self.base_row_v=nn.Conv2d(1,hs,kernel_size=(1,input_shape[0][-1]),stride=1)
        #self.base_col_v=nn.Conv2d(1,hs,kernel_size=(input_shape[0][2],1),stride=1)
        #self.base_all=nn.Conv2d(1,3*hs,kernel_size=(input_shape[0][2],input_shape[0][-1]),stride=1)
        conv_out_size=self._get_conv_out(input_shape)+input_shape[1][1]
        self.tanh=tanh
        self.fc_p=nn.Sequential(
            nn.PReLU(),
            nn.Linear(conv_out_size,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs))

        if self.depart:
            self.base_row_v=deepcopy(self.base_row_p)
            self.base_col_v=deepcopy(self.base_col_p)
            self.fc_v=deepcopy(self.fc_p)

        if self.tanh:
            F=lambda x,y:nn.Sequential(
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(hs,y),
                nn.Tanh())
        else:
            F=lambda x,y:nn.Sequential(
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(x,hs),
                nn.PReLU(),
                nn.Linear(hs,y))
        self.critic_out=nn.Sequential(
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,hs),
            nn.PReLU(),
            nn.Linear(hs,1))
        #self.policy_layer=nn.ModuleList([F(hs+self.num_processors,input_shape[0][2]) for _ in range(num_subtasks)])
        self.policy_layer=nn.ModuleList([F(hs,input_shape[0][2]) for _ in range(num_subtasks)])
        self.prior_layer=nn.ModuleList([F(hs,num_subtasks) for _ in range(num_subtasks-1)])
    
    def _get_conv_out(self,input_shape):
        s=input_shape[0][-2:]
        o_row=self.base_row_p(torch.zeros(1,1,*s))
        o_col=self.base_col_p(torch.zeros(1,1,*s))
        #o_all=self.base_all(torch.zeros(1,1,*s))
        #return int(np.prod(o_row.shape)+np.prod(o_col.shape)+np.prod(o_all.shape))
        return int(np.prod(o_row.shape)+np.prod(o_col.shape))
    
    def forward(self,x):
        conv_out_row=self.base_row_p(x[0]).view(x[0].size()[0],-1)
        conv_out_col=self.base_col_p(x[0]).view(x[0].size()[0],-1)
        conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
        out_fc=self.fc_p(conv_out)
        l1=[]
        for layer,i in zip(self.policy_layer,range(self.num_subtasks)):
            u=x[0][:,:,:,-self.num_subtasks+i].view(-1,self.num_processors)
            #v=torch.cat((out_fc,u),dim=1)
            #w=layer(v)
            if self.tanh:
                w=3*layer(out_fc)
            else:
                w=layer(out_fc)
            p=(1/(-u))+1+w
            if p.sum().isnan().item():
                print('net_here')
            l1.append(p)
        l2=[3*layer(out_fc) if self.tanh else layer(out_fc)  for layer in self.prior_layer]
        if self.depart:
            conv_out_row=self.base_row_v(x[0]).view(x[0].size()[0],-1)
            conv_out_col=self.base_col_v(x[0]).view(x[0].size()[0],-1)
            conv_out=torch.cat((conv_out_row,conv_out_col,x[1]),1)
            out_fc=self.fc_v(conv_out)
        critic=self.critic_out(out_fc)
        return (l1,l2),critic

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="prelu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)