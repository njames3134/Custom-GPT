import torch
import torch.nn as nn

block_size = 256 # context length
batch_size = 16
buffer_size = 1024 * 1024
device = 'cpu'

class DataLoader():
    def __init__(self, file_path, buffer_size, block_size, batch_size):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.file = open(file_path, 'r', encoding='utf-8')
        self.buffer = ''
        self.end_of_file = False
        self.chars = self._get_chars()
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi.get(c, 0) for c in s] # to integers
        self.decode = lambda l: ''.join([self.itos[i] for i in l]) # to string
        self._fill_buffer()

    def _get_chars(self):
        chars = set()
        while not self.end_of_file:
            chunk = self.file.read(self.buffer_size)
            if not chunk:
                self.end_of_file = True
            chars.update(chunk)
        self.file.seek(0)
        self.end_of_file = False
        return sorted(chars)

    def _fill_buffer(self):
        self.buffer = ''
        chunk = self.file.read(self.buffer_size)
        if not chunk:
            self.end_of_file = True
        self.buffer += chunk

    def get_batch(self):
        data = torch.tensor(self.encode(self.buffer), dtype=torch.long)
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        self.buffer = self.buffer[self.block_size:]  # Remove used data
        self._fill_buffer() # replace it with the next continuous buffer

        return x, y

    def close(self):
        self.file.close()

data_loader = DataLoader('wikipedia-dump.txt', buffer_size, block_size, batch_size)
x, y = data_loader.get_batch()
