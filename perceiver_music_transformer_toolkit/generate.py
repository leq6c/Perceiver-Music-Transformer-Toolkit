import torch

def generate_continue(model, inputs, number_of_prime_tokens=512, number_of_tokens_to_generate=512, temperature=0.8):
    #@markdown NOTE: Play with the settings to get different results
    # number_of_prime_tokens = 512 #@param {type:"slider", min:16, max:512, step:16}
    # number_of_tokens_to_generate = 512 #@param {type:"slider", min:64, max:512, step:32}
    # temperature = 0.8 #@param {type:"slider", min:0.1, max:1, step:0.1}

    #===================================================================
    print('=' * 70)
    print('Perceiver Music Model Continuation Generator')
    print('=' * 70)

    print('Generation settings:')
    print('=' * 70)
    print('Number of prime tokens:', number_of_prime_tokens)
    print('Number of tokens to generate:', number_of_tokens_to_generate)
    print('Model temperature:', temperature)

    print('=' * 70)
    print('Generating...')

    inp = [0, 127+128, 127+256, 0+384] * 8192

    inp = inp[:-(number_of_prime_tokens+len(inputs[:number_of_prime_tokens]))] + inputs[:number_of_prime_tokens]

    inp = torch.LongTensor(inp).cuda()

    out = model.generate(inp[None, ...], 
                         number_of_tokens_to_generate, 
                         temperature=temperature)  

    out1 = out.cpu().tolist()[0]
    return out1