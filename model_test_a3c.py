import CS_ENV
import torch
from TEST import model_test
from CS_A3C_test import env_c,f_worker,MAXNUM_TASKS,W,TANH,DEVICE
import AGENT_NET

print('start_test'+'#'*60)
f_worker.agent=AGENT_NET.DoubleNet_softmax_simple(W,MAXNUM_TASKS,TANH).to(DEVICE)
f_worker.agent.load_state_dict(torch.load("./data/CS_A3C_model_parameter.pkl"))
tl_0=model_test(env_c,f_worker,50)
print('#'*20)
env_c.cut_states=False
r_agent=CS_ENV.OTHER_AGENT(CS_ENV.random_choice,MAXNUM_TASKS)
tl_1=model_test(env_c,r_agent,50)
print('#'*20)
s_agent=CS_ENV.OTHER_AGENT(CS_ENV.short_twe_choice,MAXNUM_TASKS)
tl_2=model_test(env_c,s_agent,50)
print('agent_choice:{},r_choice:{},short_wait_choice:{}'.format(tl_0,tl_1,tl_2))