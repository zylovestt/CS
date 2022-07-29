import numpy as np

def model_test(env,agent,num_episodes):
    #env.train=False
    env.set_test_mode()
    return_list = []
    #num_subtasks=env.num_subtasks
    #np.random.seed(seed)
    for _ in range(num_episodes):
        episode_return = 0
        state = env.reset()
        done = 0
        while not done:
            #print('sub_loc:\n{}'.format(state[0][0,0,:,-num_subtasks:]))
            action = agent.take_action(state)
            #print('action0:{}\naction1:{}'.format(action[0],action[1]))
            next_state, reward, done,*_ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        '''if (i+1)%cycles==0:
            print('episode:{}, reward:{}'.format(i+1,np.mean(return_list[-cycles])))'''
    env.set_train_mode()
    #env.reset()
    return np.mean(return_list)