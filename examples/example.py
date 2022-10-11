from perceiver_music_transformer_toolkit.PerceiverMusicTransformerToolkit import PerceiverMusicTransformerToolkit
from perceiver_music_transformer_toolkit.load_train_data import load_train_data
from perceiver_music_transformer_toolkit.midi_to_input import create_dataset, midifile_to_input
from perceiver_music_transformer_toolkit.input_to_midi import input_to_midi
from perceiver_music_transformer_toolkit.toolkit_io import save_toolkit_params, load_toolkit_params

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
pmt = PerceiverMusicTransformerToolkit(SEQ_LEN=128, DEPTH=1, DIMS=128, PREFIX_SEQ_LEN=64)
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
