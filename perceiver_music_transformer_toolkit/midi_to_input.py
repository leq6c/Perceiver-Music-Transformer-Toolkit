import os
from tqdm import tqdm
import random
import secrets
from tegridy_tools import TMIDIX
import matplotlib.pyplot as plt
from perceiver_music_transformer_toolkit.default_patch_map import default_patch_map, default_patch_map_names

# Code referenced from https://github.com/asigalov61/Euterpe

def midi_to_input(binary_data, patch_map=default_patch_map, stats=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
    melody_chords_f = []
    score = TMIDIX.midi2ms_score(binary_data)

    events_matrix = []

    itrack = 1

    patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    while itrack < len(score):
        for event in score[itrack]:         
            if event[0] == 'note' or event[0] == 'patch_change':
                events_matrix.append(event)
        itrack += 1

    events_matrix.sort(key=lambda x: x[1])

    events_matrix1 = []
    for event in events_matrix:
            if event[0] == 'patch_change':
                patches[event[2]] = event[3]

            if event[0] == 'note':
                event.extend([patches[event[3]]])
                once = False

                for p in patch_map:
                    if event[6] in p and event[3] != 9: # Except the drums
                        event[3] = patch_map.index(p)
                        once = True

                if not once and event[3] != 9: # Except the drums
                    event[3] = 0 # All other instruments/patches channel
                    event[5] = max(80, event[5])

                if event[3] < 12: # We won't write chans 11-16 for now...
                    events_matrix1.append(event)
                    stats[event[3]] += 1

    # Sorting...
    events_matrix1.sort(key=lambda x: (x[1], x[3]))

    # recalculating timings
    for e in events_matrix1:
        e[1] = int(e[1] / 16)
        e[2] = int(e[2] / 32)

    # final processing...

    pe = events_matrix1[0]
    for e in events_matrix1:

        time = max(0, min(127, e[1]-pe[1]))
        dur = max(1, min(127, e[2]))
        cha = max(0, min(11, e[3]))
        ptc = max(1, min(127, e[4]))
        vel = max(19, min(127, e[5]))

        div_vel = int(vel / 19)

        chan_vel = (cha * 11) + div_vel

        melody_chords_f.extend([chan_vel, time+128, dur+256, ptc+384])

        pe = e
    
    return melody_chords_f

def midifile_to_input(path, patch_map=default_patch_map):
    return midi_to_input(open(path, "rb").read(), patch_map)

def create_dataset(input_directory, output_directory, patch_map=default_patch_map, patch_map_names=default_patch_map_names):
    filez = list()
    for (dirpath, dirnames, filenames) in os.walk(input_directory):
        filez += [os.path.join(dirpath, file) for file in filenames]
    print('=' * 70)

    if filez == []:
        print('Could not find any MIDI files. Please check Dataset dir...')
        print('=' * 70)

    print('Randomizing file list...')
    random.shuffle(filez)
    
    sorted_or_random_file_loading_order = False # Sorted order is NOT usually recommended
    dataset_ratio = 1 # Change this if you need more data


    print('TMIDIX MIDI Processor')
    print('Starting up...')
    ###########

    files_count = 0

    gfiles = []

    melody_chords_f = []

    stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    output_file_begins = os.path.join(output_directory, "INTs_")

    print('Processing MIDI files. Please wait...')
    for f in tqdm(filez[:int(len(filez) * dataset_ratio)]):
        try:
            fn = os.path.basename(f)
            fn1 = fn.split('.')[0]

            #print('Loading MIDI file...')
            input_data = midi_to_input(open(f, 'rb').read(), patch_map, stats)
            
            melody_chords_f.extend(input_data)

            # Break between compositions
            melody_chords_f.extend([0, 127+128, 127+256, 0+384])
            melody_chords_f.extend([0, 127+128, 127+256, 0+384])

            files_count += 1

            if files_count % 4000 == 0:
              count = str(files_count)
              TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, output_file_begins + count)
              melody_chords_f = []

        except KeyboardInterrupt:
            print('Saving current progress and quitting...')
            break  

        except:
            print('Bad MIDI:', f)
            continue

    count = str(files_count)
    TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, output_file_begins + count)

    print('=' * 70)

    print('Done!')   
    print('=' * 70)

    print('Resulting Stats:')
    print('=' * 70)
    print('Total MIDI Files:', files_count)
    print('=' * 70)

    for i in range(len(patch_map_names)):
        print(patch_map_names[i] + ":", stats[i])

    print('=' * 70)
