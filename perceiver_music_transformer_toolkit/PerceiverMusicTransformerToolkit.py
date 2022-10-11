from perceiver_music_transformer_toolkit.create_model import create_model, load_model
from perceiver_music_transformer_toolkit.prepare_train_data import prepare_train_data
from perceiver_music_transformer_toolkit.train import train
from perceiver_music_transformer_toolkit.generate import generate_continue

class PerceiverMusicTransformerToolkit:
    def __init__(self, 
                 SEQ_LEN=8192*4, 
                 PREFIX_SEQ_LEN=(8192*4)-1024,
                 BATCH_SIZE=4,
                 GRADIENT_ACCUMULATE_EVERY=4,
                 LEARNING_RATE=2e-4,
                 VALIDATE_EVERY=100,
                 GENERATE_EVERY=200,
                 SAVE_EVERY=2000,
                 GENERATE_LENGTH=32,
                 NUM_TOKENS=512,
                 HEADS=16,
                 DIM_HEAD=64,
                 CROSS_ATTN_DROPOUT=0.5,
                 DIMS=1024,
                 DEPTH=24,
                 NUM_BATCHES=-1):
        self.SEQ_LEN = SEQ_LEN
        self.PREFIX_SEQ_LEN = PREFIX_SEQ_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.GRADIENT_ACCUMULATE_EVERY = GRADIENT_ACCUMULATE_EVERY
        self.LEARNING_RATE = LEARNING_RATE
        self.VALIDATE_EVERY = VALIDATE_EVERY
        self.GENERATE_EVERY = GENERATE_EVERY
        self.SAVE_EVERY = SAVE_EVERY
        self.GENERATE_LENGTH = GENERATE_LENGTH
        self.NUM_TOKENS = NUM_TOKENS
        self.HEADS = HEADS
        self.DIM_HEAD = DIM_HEAD
        self.CROSS_ATTN_DROPOUT = CROSS_ATTN_DROPOUT
        self.DIMS = DIMS
        self.DEPTH = DEPTH
        self.NUM_BATCHES = NUM_BATCHES

    def init_model(self):
        model, optim = create_model(self.NUM_TOKENS, self.DIMS, self.DEPTH, self.HEADS, self.DIM_HEAD, self.CROSS_ATTN_DROPOUT, self.SEQ_LEN, self.PREFIX_SEQ_LEN, self.NUM_BATCHES, self.LEARNING_RATE)
        self.model = model
        self.optim = optim
    
    def prepare_train_data(self, train_data):
        self.NUM_BATCHES = len(train_data) // self.SEQ_LEN // self.BATCH_SIZE
        return prepare_train_data(train_data, self.SEQ_LEN, self.BATCH_SIZE)
    
    def train(self, train_loader, val_loader, checkpoint_dir, enable_plt=False):
        train(self.model, self.optim, train_loader, val_loader, self.NUM_BATCHES, self.GRADIENT_ACCUMULATE_EVERY, self.VALIDATE_EVERY, self.GENERATE_EVERY, self.GENERATE_LENGTH, self.SAVE_EVERY, checkpoint_dir, enable_plt=enable_plt)
    
    def load_model(self, full_path_to_model_checkpoint):
        model, optim = load_model(self.NUM_TOKENS, self.DIMS, self.DEPTH, self.HEADS, self.DIM_HEAD, self.CROSS_ATTN_DROPOUT, self.SEQ_LEN, self.PREFIX_SEQ_LEN, self.NUM_BATCHES, self.LEARNING_RATE)
        self.model = model
        self.optim = optim
    
    def eval(self):
        self.model.eval()
    
    def generate_continue(self, inputs, number_of_prime_tokens=512, number_of_tokens_to_generate=512, temperature=0.8):
        return generate_continue(self.model, inputs, number_of_prime_tokens, number_of_tokens_to_generate, temperature)
