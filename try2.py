import torch.multiprocessing as mp
import torch

class G:
    ga=torch.tensor(5,dtype=torch.float32,requires_grad=True)
    y=torch.tensor(1,dtype=torch.float32)
    ga.share_memory_()
    y.share_memory_()
    def f(self):
        self.y+=self.ga**2
        self.y.backward()
        self.y+=1

if __name__=='__main__':
    #mp.set_start_method('spawn', force=True)
    g=G()
    p1=mp.Process(target=g.f)
    p1.start()
    print(g.y)
    p1.join()