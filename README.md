# Perceiver Music Transformer Toolkit
Toolset for easy development with Perceive-Music-Transformer (https://github.com/asigalov61/Perceiver-Music-Transformer). Only supports multi-instrumental, currently. 

# Paper
- https://arxiv.org/abs/2202.07765

# Note for prameters tuning
- `SEQ_LEN`: Perceiver-AR inputs are fixed length. You can specify max length of its sequence. 
- `PREFIX_SEQ_LEN`: Length of Key and Value for cross-attention.
- `HEADS`: Number of heads of multi-head-attention. 
- `DEPTH`: Number of latents-self-attention. 

# Links
- https://github.com/asigalov61/Perceiver-Music-Transformer
- https://github.com/asigalov61/Euterpe
- https://github.com/asigalov61/tegridy-tools

# Example
- Create dataset
```
create_dataset("mids", "dataset")
```

- load dataset
```
train_data = load_train_data("dataset")
```

- train
```
pmt = PerceiverMusicTransformerToolkit() # you can specify params here to train
pmt.init_model()

# prepare training data
train_loader, val_loader = pmt.prepare_train_data(train_data)

# train
pmt.train(train_loader, val_loader, "checkpoints")
```

- eval
```
# generate continue
input_data = midifile_to_input("seed.mid")
output_data = pmt.generate_continue(input_data)

# output to midi
input_to_midi(output_data, "output.mid")

# save toolkit params
save_toolkit_params(pmt, "params.dump")
```

- load
```
pmt = load_toolkit_params("params.dump")
# load ckpt
pmt.load_model("checkpoints/...")
```
