import numpy as np

class Advque:
    def __init__(self, size=50000):
        self.size = size 
        self.current_size = 0
        self.que = np.zeros(size)
        self.idx = 0
    
    def update(self, values):
        l = len(values)

        if self.idx + l <= self.size:
            idxes = np.arange(self.idx, self.idx+l)
        else:
            idx1 = np.arange(self.idx, self.size)
            idx2 = np.arange(0, self.idx+l -self.size)
            idxes = np.concatenate((idx1, idx2))
        self.que[idxes] = values.reshape(-1)

        self.idx = (self.idx + l) % self.size 
        self.current_size = min(self.current_size+l, self.size)

    def get(self, threshold):
        return np.percentile(self.que[:self.current_size], threshold)

advque = Advque()


        
        
