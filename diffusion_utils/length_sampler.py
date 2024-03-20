import numpy as np
from collections import Counter
from utils.preprocessing import load_fasta_file

class LengthSampler:
    def __init__(self, path, max_len=254):
        data = load_fasta_file(path)
        self.dataset_len = np.clip([len(t) for t in data], a_min=0, a_max=max_len)
        freqs = Counter(self.dataset_len)
        self.distrib = []
        for i in range(max_len + 1):
            self.distrib.append(freqs.get(i, 0))
            
        self.distrib = np.array(self.distrib) / np.sum(self.distrib)
            
    def sample(self, num_samples):
        s = np.argmax(np.random.multinomial(1, self.distrib, size=(num_samples)), axis=1)
        return s
    
class CustomLengthSampler:
    def __init__(self, data, max_len=254):
        self.dataset_len = np.clip([len(t[0]) for t in data], a_min=0, a_max=max_len)
        self.dataset_contexts = [t[1] for t in data]
        freqs = Counter(self.dataset_len)
        freqs_contexts = Counter(self.dataset_contexts)
        self.distrib = []
        for i in range(max_len + 1):
            self.distrib.append(freqs.get(i, 0))
        self.context_distrib = []
        for i in list(set(self.dataset_contexts)):
            self.context_distrib.append(freqs_contexts.get(i, 0))
        self.distrib = np.array(self.distrib) / np.sum(self.distrib)
        self.context_distrib = np.array(self.context_distrib) / np.sum(self.context_distrib)
            
    def sample(self, num_samples):
        s = np.argmax(np.random.multinomial(1, self.distrib, size=(num_samples)), axis=1)
        c = np.argmax(np.random.multinomial(1, self.context_distrib, size=(num_samples)), axis=1)
        return s, c