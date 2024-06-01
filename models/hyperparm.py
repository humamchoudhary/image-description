# import os
# import sys

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# print(parent_dir)
# sys.path.insert(0, parent_dir)

from models.const import *

n_epochs = 30
batch_size = 32
# output_dim = len(vocab)
embedding_dim = 256
encoder_dim = 512
decoder_dim = 512
attention_dim = 512 * 3
decoder_dropout = 0.3


teacher_forcing_ratio = 0.6
