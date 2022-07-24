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

'''class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l=nn.Linear(1,1)

device = "cuda" 
net=NN().to(device)
print(net)
net.share_memory()'''

import torch.multiprocessing as mp
import torch

def foo(worker,tl):
    tl[worker] += (worker+1) * 1000

if __name__ == '__main__':
    #mp.set_start_method('spawn', force=True)
    tl = [torch.randn(2), torch.randn(3)]

    for t in tl:
        t.share_memory_()

    print("before mp: tl=")
    print(tl)

    p0 = mp.Process(target=foo, args=(0, tl))
    p1 = mp.Process(target=foo, args=(1, tl))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("after mp: tl=")
    print(tl)