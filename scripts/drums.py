import os
from essentia.standard import *
from essentia import Pool, array
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
from scipy.io.wavfile import read, write

def get_time_steps_from_annotations(annotations):
    measures = annotations['interpretation']
    beats = annotations['beats']
    measure_duration = int(annotations['meter'].split('/')[0])
    beat_figure = int(annotations['meter'].split('/')[1])
    idx_step=0
    inter_idx_step = 0
    time_steps = []
    notes = []
    silence_count = 0
    nothing_annotated = True
    for i, measure in enumerate(measures):
        if type(measure) is list:
            silence_count = silence_count + 1
            continue
        events = measure.split(' ')
        for event in enumerate(events):
            note, duration = event[1].split(':')
            if i - silence_count == 0:
                if note == 'X' and nothing_annotated:
                    continue
                else:
                    nothing_annotated = False
            duration = int(duration)
            time_steps.append( beats[idx_step] + inter_idx_step*(beats[idx_step + 1] - beats[idx_step]))
            notes.append(note)
            if duration > beat_figure:
                inter_idx_step = inter_idx_step + beat_figure/duration
                if inter_idx_step >= 1:
                    inter_idx_step = 0
                    idx_step = idx_step + 1
            else:
                idx_step = idx_step + int(beat_figure/duration)
                inter_idx_step = 0
    return notes, np.array(time_steps), np.array(beats)           

def SliceDrums_BeatDetection(folder, audio_filename, fs):
    od_hfc = OnsetDetection(method = 'hfc')
    w = Windowing(type = 'hann')
    fft = FFT() # this gives us a complex FFT
    c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)
    onsets = Onsets()

    x = MonoLoader(filename = folder + audio_filename, sampleRate = fs)()
    duration = float(len(x)) / fs

    x = x / np.max(np.abs(x))
    
    t = np.arange(len(x)) / float(fs)
    
    zero_array = t * 0 #used only for plotting purposes

    #Plotting
    f, axarr = plt.subplots(1,1,figsize=(80, 20))
    

    #Essentia beat tracking
    pool = Pool()
    for frame in FrameGenerator(x, frameSize = 1024, hopSize = 512):
        mag, phase, = c2p(fft(w(frame)))
        pool.add('features.hfc', od_hfc(mag, phase))


    onsets_list = onsets(array([pool['features.hfc']]), [1])
    axarr.vlines(onsets_list , -1, 1, color = 'k',zorder=2, linewidth=5.0)
    axarr.plot(t,x,zorder=1)
    axarr.axis('off')
    for i, onset in enumerate(onsets_list):
        sample = int(onset * fs) - 1000
        samplename =  "{}slices/{}{}__blind.wav".format(folder, str(len(str(i))), str(i))
        if(i >=  len(onsets_list)-1):
            next_sample = len(x) 
        else:
            next_sample = int(onsets_list[i+1]*fs) - 1000
        x_seg = x[sample  :  next_sample]
        MonoWriter(filename=samplename)(x_seg)
        
    return onsets_list, duration

def assess_notes(events, y_pred_):
    notes_correct = np.array([i==j for i, j in zip(events, y_pred_)]) * 1.0
    for i, note in enumerate(notes_correct):
      if note == 0:
          pred = y_pred_[i].split('+')
          for p in pred:
            if p in events[i].split('+') and p != 'hh':
                notes_correct[i] = 0.75
    return notes_correct

def map_onsets_with_events(onsets_detected, onsets_expected, y_pred):
    distance=60.0
    mapped = ['no']*len(onsets_expected)
    for onset, predicted in zip(onsets_detected, y_pred):
      for i, expected in enumerate(onsets_expected):
          if abs(onset - expected) < distance:
              distance = abs(onset - expected)
          else:
              mapped[i-1] = predicted
              distance=60.0
              break   
    if abs(onset - expected) < 0.01:
        mapped[-1] = predicted

    return mapped
