from tegridy_tools import TMIDIX
from perceiver_music_transformer_toolkit.default_patch_map import default_list_of_midi_patches

def input_to_song(inputs):
    song = inputs
    song_f = []
    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0
    son = []
    song1 = []

    for s in song:
      if s > 127:
        son.append(s)

      else:
        if len(son) == 4:
          song1.append(son)
        son = []
        son.append(s)
    
    for s in song1:
      if s[0] > 0 and s[1] >= 128:
        if s[2] > 256 and s[3] > 384:

          channel = s[0] // 11

          vel = (s[0] % 11) * 19

          time += (s[1]-128) * 16
      
          dur = (s[2] - 256) * 32
          
          pitch = (s[3] - 384)
                                    
          song_f.append(['note', time, dur, channel, pitch, vel ])
    
    return song_f

def song_to_midi(song_f, output_path, list_of_midi_patches=default_list_of_midi_patches, signature="Perceiver", track_name="Untitled", number_of_ticks_per_quarter=500):
    return TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                        output_signature = signature,  
                                        output_file_name = output_path, 
                                        track_name=track_name,
                                        list_of_MIDI_patches=list_of_midi_patches,
                                        number_of_ticks_per_quarter=number_of_ticks_per_quarter)

def input_to_midi(inputs, output_path, list_of_midi_patches=default_list_of_midi_patches, signature="Perceiver", track_name="Untitled", number_of_ticks_per_quarter=500):
    return song_to_midi(input_to_song(inputs), output_path, list_of_midi_patches, signature, track_name, number_of_ticks_per_quarter)
