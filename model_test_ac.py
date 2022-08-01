import CS_ENV
import torch
from TEST import model_test
from CS_AC_test import env_c,agent,maxnum_tasks

print('start_test'+'#'*60)
agent.agent.load_state_dict(torch.load("./data/CS_AC_model_parameter.pkl"))
tl_0=model_test(env_c,agent,10)
print('#'*20)
env_c.cut_states=False
r_agent=CS_ENV.OTHER_AGENT(CS_ENV.random_choice,maxnum_tasks)
tl_1=model_test(env_c,r_agent,10)
print('#'*20)
s_agent=CS_ENV.OTHER_AGENT(CS_ENV.short_twe_choice,maxnum_tasks)
tl_2=model_test(env_c,s_agent,10)
print('agent_choice:{},r_choice:{},short_wait_choice:{}'.format(tl_0,tl_1,tl_2))