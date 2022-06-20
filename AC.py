import torch
import torch.nn.functional as FU
import torch.nn.utils as nn_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import AGENT_NET

class ActorCritic:
    def __init__(self,input_shape:tuple,num_subtasks,actor_lr,critic_lr,gamma,device,clip_grad,beta,conv):
        if conv:
            self.actor=AGENT_NET.PolicyNet(input_shape,num_subtasks).to(device)
            self.critic=AGENT_NET.ValueNet(input_shape,num_subtasks).to(device)
        else:
            self.actor=AGENT_NET.PolicyNet_FC(input_shape,num_subtasks).to(device)
            self.critic=AGENT_NET.ValueNet_FC(input_shape,num_subtasks).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        #self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        #self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.beta=beta
        self.clip_grad=clip_grad
        self.device=device
    
    def take_action(self,state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        probs_subtasks_orginal,probs_prior_orginal=self.actor(state)
        '''probs_subtasks_orginal*=[x*y
            for x,y in zip(probs_subtasks_orginal,state[0][0,0,:,-self.num_subtasks:].T)]'''
        action_subtasks=[torch.distributions.Categorical(x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action
    
    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        dones=F(transition_dict['dones']).view(-1,1)

        # 时序差分目标
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)  # 时序差分误差
        probs=self.actor(states)
        s=0
        #probs=(probs[0]+1e-10,probs[1]+1e-10)
        for prob in probs[0]:
            s+=(prob*prob.log()).sum(dim=1)
        t=0
        for prob in probs[1]:
            t+=(prob*prob.log()).sum(dim=1)
        u=self.beta*(s.mean()+t.mean())
        log_probs=torch.log(self.calculate_probs(probs,actions))
        actor_loss=torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss=torch.mean(FU.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        #print(actor_loss)
        #print(critic_loss)
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        nn_utils.clip_grad_norm_(self.actor.parameters(),self.clip_grad)
        nn_utils.clip_grad_norm_(self.critic.parameters(),self.clip_grad)
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
    
    def calculate_probs(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[0][i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        probs=F(0)

        G=lambda i:((torch.gather(out_puts[1][i],1,actions[1][:,[i]])+1e-7)
            /(out_puts[1][i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True)+1e-7)*G(i+1)
            if i<self.num_subtasks else 1.0)
        probs*=G(0)
        return probs

class ACTWOSTEPS(ActorCritic):
    def __init__(self, input_shape: tuple, num_subtasks, actorf_lr, actors_lr, critic_lr, gamma, device, clip_grad, beta):
        self.actorf=AGENT_NET.PolicyNet_First(input_shape,num_subtasks).to(device)
        self.actors=AGENT_NET.PolicyNet_Second(input_shape,num_subtasks).to(device)
        self.critic=AGENT_NET.ValueNet(input_shape,num_subtasks).to(device)
        self.actorf_optimizer=torch.optim.Adam(self.actorf.parameters(),lr=actorf_lr,eps=1e-3)
        self.actors_optimizer=torch.optim.Adam(self.actors.parameters(),lr=actors_lr,eps=1e-3)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr,eps=1e-3)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.beta=beta
        self.clip_grad=clip_grad
        self.device=device

    def take_action(self,state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        probs_subtasks_orginal=self.actorf(state)
        probs_prior_orginal=self.actors(state)
        action_subtasks=[torch.distributions.Categorical(x).sample().item() for x in probs_subtasks_orginal]
        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action

    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))
        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        dones=F(transition_dict['dones']).view(-1,1)

        # 时序差分目标
        td_target=rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta=td_target-self.critic(states)  # 时序差分误差
        probs_subtasks_orginal=self.actorf(states)
        probs_prior_orginal=self.actors(states)
        F=lambda i,x:(x[i]*x[i].log()).sum(dim=1)+F(i+1,x) if i<len(x) else 0
        f=self.beta*F(0,probs_subtasks_orginal).mean()
        s=self.beta*F(0,probs_prior_orginal).mean()
        log_probs_f=torch.log(self.calculate_probs_f(probs_subtasks_orginal,actions))
        actor_loss_f=torch.mean(-log_probs_f * td_delta.detach())+f
        log_probs_s=torch.log(self.calculate_probs_s(probs_prior_orginal,actions))
        actor_loss_s=torch.mean(-log_probs_s * td_delta.detach())+s
        # 均方误差损失函数
        critic_loss=torch.mean(FU.mse_loss(self.critic(states), td_target.detach()))
        self.actorf_optimizer.zero_grad()
        self.actors_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        #print(actor_loss)
        #print(critic_loss)
        actor_loss_f.backward()  # 计算策略网络的梯度
        actor_loss_s.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        nn_utils.clip_grad_norm_(self.actorf.parameters(),self.clip_grad)
        nn_utils.clip_grad_norm_(self.actors.parameters(),self.clip_grad)
        nn_utils.clip_grad_norm_(self.critic.parameters(),self.clip_grad)
        self.actorf_optimizer.step()  # 更新策略网络的参数
        self.actors_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
    
    def calculate_probs_f(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        return F(0)
    
    def calculate_probs_s(self,out_puts,actions):
        F=lambda i:(torch.gather(out_puts[i],1,actions[1][:,[i]])
            /(out_puts[i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[i],1,actions[1][:,:i]).sum(axis=1,keepdim=True))*F(i+1)
            if i<self.num_subtasks else 1.0)
        return F(0)
    
class ActorCritic_Double:
    def __init__(self,input_shape:tuple,num_subtasks,lr,weights,gamma,device,clip_grad,beta,n_steps,mode,labda):
        self.writer=SummaryWriter()
        self.step=0
        self.agent=AGENT_NET.DoubleNet(input_shape,num_subtasks).to(device)
        #self.agent_optimizer=torch.optim.Adam(self.agent.parameters(),lr=lr,eps=1e-3)
        self.agent_optimizer=torch.optim.SGD(self.agent.parameters(),lr=lr,momentum=0.9)
        self.input_shape=input_shape
        self.num_processors=input_shape[0]
        self.num_subtasks=num_subtasks
        self.gamma=gamma
        self.beta=beta
        self.clip_grad=clip_grad
        self.weights=weights
        self.n_steps=n_steps
        self.device=device
        self.mode=mode
        if mode=='gce':
            self.labda=labda
        self.cri_loss=[]
        self.act_loss=[]
        self.eposub_loss=[]
        self.epopri_loss=[]
        self.agent_loss=[]
        self.ac_loss=[]
    
    def take_action(self,state):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        state=(F(state[0]),F(state[1]))
        u=state[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in state[1]:
            i[:]=(i-i.mean())/i.std()
        (probs_subtasks_orginal,probs_prior_orginal),_=self.agent(state)
        '''probs_subtasks_orginal*=[x*y
            for x,y in zip(probs_subtasks_orginal,state[0][0,0,:,-self.num_subtasks:].T)]'''
        action_subtasks=[torch.distributions.Categorical(x).sample().item()
            for x in probs_subtasks_orginal]

        action_prior=[]
        probs_prior_orginal=torch.cat(probs_prior_orginal,0)
        probs_prior_orginal=torch.tensor(
            np.concatenate((probs_prior_orginal.cpu().detach().numpy(),np.arange(self.num_subtasks).reshape(1,-1)),0),dtype=torch.float)
        for i in range(self.num_subtasks):
            x=torch.tensor(np.delete(probs_prior_orginal.numpy(),action_prior,1),dtype=torch.float)
            action_prior.append(x[-1,torch.distributions.Categorical(x[i]).sample().item()].int().item())
        
        action=np.zeros((2,self.num_subtasks),dtype='int')
        action[0]=action_subtasks
        action[1]=action_prior
        return action
    
    def update(self, transition_dict:dict):
        F=lambda x:torch.tensor(x,dtype=torch.float).to(self.device)
        states=tuple(F(np.concatenate([x[i] for x in transition_dict['states']],0)) for i in range(len(transition_dict['states'][0])))
        u=states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in states[1]:
            i[:]=(i-i.mean())/i.std()
        actions=tuple(F(np.vstack([x[i] for x in transition_dict['actions']])).type(torch.int64) for i in range(len(transition_dict['actions'][0])))

        rewards=F(transition_dict['rewards']).view(-1,1)
        next_states=tuple(F(np.concatenate([x[i] for x in transition_dict['next_states']],0)) for i in range(len(transition_dict['states'][0])))
        u=next_states[0][:,:,:,:-self.num_subtasks]
        for i in u:
            i[:]=(i-i.mean())/i.std()
        for i in next_states[1]:
            i[:]=(i-i.mean())/i.std()
        overs=F(transition_dict['overs']).view(-1,1)
        # 时序差分目标
        #td_target=rewards+self.gamma*self.agent(next_states)[1]*(1-dones)
        if self.mode=='n_steps':
            F_td=self.cal_nsteps
        elif self.mode=='n_steps_all':
            F_td=self.cal_nsteps_all
        elif self.mode=='gce':
            F_td=self.cal_gce
        td_delta=F_td(rewards,states,next_states,overs)  # 时序差分误差
        td_target=td_delta+self.agent(states)[1]
        self.ac_loss.append(td_delta.mean().item())
        probs=self.agent(states)[0]
        s=0
        #probs=(probs[0]+1e-10,probs[1]+1e-10)
        for prob in probs[0]:
            s+=(prob*prob.log()).sum(dim=1)
        t=0
        for prob in probs[1]:
            t+=(prob*prob.log()).sum(dim=1)
        self.eposub_loss.append(s.mean().item())
        self.epopri_loss.append(t.mean().item())
        epo_loss=self.beta*(s.mean()+t.mean())
        log_probs=torch.log(self.calculate_probs(probs,actions))
        actor_loss=torch.mean(-log_probs * td_delta.detach())
        self.act_loss.append(actor_loss.item())
        # 均方误差损失函数
        critic_loss=torch.mean(FU.mse_loss(self.agent(states)[1], td_target.detach()))
        self.cri_loss.append(critic_loss.item())
        agent_loss=epo_loss+actor_loss+self.weights*critic_loss
        self.agent_loss.append(agent_loss.item())
        if torch.isnan(agent_loss)>0:
            print("here!")
        self.agent_optimizer.zero_grad()
        agent_loss.backward()
        nn_utils.clip_grad_norm_(self.agent.parameters(),self.clip_grad)
        self.agent_optimizer.step()  # 更新策略网络的参数
        self.writer.add_scalar(tag='cri_loss',scalar_value=self.cri_loss[-1],global_step=self.step)
        self.writer.add_scalar(tag='act_loss',scalar_value=self.act_loss[-1],global_step=self.step)
        self.writer.add_scalar(tag='eposub_loss',scalar_value=self.eposub_loss[-1],global_step=self.step)
        self.writer.add_scalar(tag='epopri_loss',scalar_value=self.epopri_loss[-1],global_step=self.step)
        self.writer.add_scalar(tag='ac_loss',scalar_value=self.ac_loss[-1],global_step=self.step)
        self.writer.add_scalar(tag='agent_loss',scalar_value=self.agent_loss[-1],global_step=self.step)
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in self.agent.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        self.writer.add_scalar("grad_l2", grad_means / grad_count, self.step)
        self.writer.add_scalar("grad_max", grad_max, self.step)
        probs_new=self.agent(states)[0]
        kl=0
        for i in range(2):
            for p_old,p_new in zip(probs[i],probs_new[i]):
                kl+=-((p_new/p_old).log()*p_old).sum(dim=1).mean().item()
        self.writer.add_scalar("kl", kl, self.step)
        self.step+=1
    
    def calculate_probs(self,out_puts,actions):
        F=lambda i:torch.gather(out_puts[0][i],1,actions[0][:,[i]])*F(i+1)\
            if i<self.num_subtasks else 1.0
        probs=F(0)

        G=lambda i:((torch.gather(out_puts[1][i],1,actions[1][:,[i]])+1e-7)
            /(out_puts[1][i].sum(axis=1,keepdim=True)
                -torch.gather(out_puts[1][i],1,actions[1][:,:i]).sum(axis=1,keepdim=True)+1e-7)*G(i+1)
            if i<self.num_subtasks else 1.0)
        probs*=G(0)
        return probs

    def cal_nsteps(self,rewards,states,next_states,overs):
        r=rewards
        rewards=torch.cat((rewards,torch.zeros(self.n_steps-1).to(self.device).view(-1,1)),dim=0)
        F=lambda x:torch.cat((x,x[[-1]]),dim=0)
        last_state_0=next_states[0]
        last_state_1=next_states[1]
        for _ in range(self.n_steps-1):
            last_state_0=F(last_state_0)
            last_state_1=F(last_state_1)
            overs=F(overs)
        var=(self.gamma**self.n_steps)*self.agent((last_state_0,last_state_1))[1][self.n_steps-1:]*(1-overs[self.n_steps-1:])
        for i in range(1,self.n_steps):
            r+=(self.gamma**i)*rewards[i:len(r)+i]
        return var+r-self.agent(states)[1]
    
    def cal_(self,r,init,discount):
        ret=r.clone()
        l_r=len(r)-1
        for i,t in enumerate(reversed(r)):
            init=init*discount+t
            ret[l_r-i]=init
        return ret
    
    def cal_nsteps_all(self,rewards,states,next_states,overs):
        init=self.agent((next_states[0][[-1]],next_states[1][[-1]]))[1]*(1-overs[-1])
        return self.cal_(rewards,init,self.gamma)-self.agent(states)[1]
    
    def cal_gce(self,rewards,states,next_states,overs):
        r=rewards+self.gamma*self.agent(next_states)[1]*(1-overs)-self.agent(states)[1]
        return self.cal_(r,0,self.gamma*self.labda)