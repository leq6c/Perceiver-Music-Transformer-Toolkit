# Perceiver Music Transformer Toolkit
Toolset for easy development with Perceive-Music-Transformer (https://github.com/asigalov61/Perceiver-Music-Transformer). Only supports multi-instrumental, currently. 

# Links
- https://github.com/asigalov61/Perceiver-Music-Transformer
- https://github.com/asigalov61/Euterpe
- https://github.com/asigalov61/tegridy-tools

# Example
```
# =======================
# create dataset from mid
# =======================
# create dataset from folder
create_dataset("mids", "dataset")

# load dataset
train_data = load_train_data("dataset")

# =======================
# training
# =======================
# init
pmt = PerceiverMusicTransformerToolkit() # you can specify params here to train
pmt.init_model()

# prepare training data
train_loader, val_loader = pmt.prepare_train_data(train_data)

# train
pmt.train(train_loader, val_loader, "checkpoints")

# =======================
# evaluation
# =======================
# generate continue
input_data = midifile_to_input("seed.mid")
output_data = pmt.generate_continue(input_data)

# output to midi
input_to_midi(output_data, "output.mid")

# save toolkit params
save_toolkit_params(pmt, "params.dump")

# =======================
# load toolkit from params
# =======================
pmt = load_toolkit_params("params.dump")
# load ckpt
pmt.load_model("checkpoints/...")
# ...
```
