import json
from perceiver_music_transformer_toolkit.PerceiverMusicTransformerToolkit import PerceiverMusicTransformerToolkit

def save_toolkit_params(tk, path):
    with open(path, 'w') as f:
        json.dump(tk.__dict__, f)

def load_toolkit_params(path):
    with open(path, 'w') as f:
        return json.load(f, object_hook=lambda x: PerceiveMusicTransformerToolkit(**x))
