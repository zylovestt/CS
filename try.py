import torch.multiprocessing as mp
import torch
import torch.nn as nn
import os
import time

'''#mp.set_start_method('spawn')
os.environ['OMP_NUM_THREADS'] = "1"

train_queue=mp.Queue(10)

class AA:
    def __init__(self,train_queue):
        self.train_queue=train_queue
    def f(self):
        for i in range(10):
            self.train_queue.put(i)
            print(i)
            time.sleep(0.1)
a1=AA(train_queue)
a2=AA(train_queue)
dac=mp.Process(target=a1.f)
dac2=mp.Process(target=a2.f,args=())
dac.start()
dac2.start()
dac.join()
dac2.join()'''

class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l=nn.Linear(1,1)

device = "cuda" 
net=NN().to(device)
print(net)
net.share_memory()