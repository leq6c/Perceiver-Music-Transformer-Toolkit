import torch
from torchsummary import summary

from tegridy_tools.perceiver_ar.perceiver_ar_pytorch import PerceiverAR
from tegridy_tools.perceiver_ar.autoregressive_wrapper import AutoregressiveWrapper

def create_model(NUM_TOKENS, DIMS, DEPTH, HEADS, DIM_HEAD, CROSS_ATTN_DROPOUT, SEQ_LEN, PREFIX_SEQ_LEN, NUM_BATCHES, LEARNING_RATE):
    # Setup model
    # instantiate model

    model = PerceiverAR(
        num_tokens = NUM_TOKENS,
        dim = DIMS,
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD,
        cross_attn_dropout = CROSS_ATTN_DROPOUT,
        max_seq_len = SEQ_LEN,
        cross_attn_seq_len = PREFIX_SEQ_LEN
    )

    model = AutoregressiveWrapper(model)
    model.cuda()

    print('Done!')

    # summary(model)
    
    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    return model, optim

def load_model(NUM_TOKENS, DIMS, DEPTH, HEADS, DIM_HEAD, CROSS_ATTN_DROPOUT, SEQ_LEN, PREFIX_SEQ_LEN, NUM_BATCHES, LEARNING_RATE):
    model, optim = create_model(NUM_TOKENS, DIMS, DEPTH, HEADS, DIM_HEAD, CROSS_ATTN_DROPOUT, SEQ_LEN, PREFIX_SEQ_LEN, NUM_BATCHES, LEARNING_RATE)
    
    state_dict = torch.load(full_path_to_model_checkpoint)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, optim
