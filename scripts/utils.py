""" Some useful utilites to load files, getting paths and processing the dataset
"""
import os
from urllib.request import urlopen
import numpy as np
import essentia.standard as ess
from essentia.standard import *
from essentia import Pool, array

def sort_string_elements(x):
    in_list = x.split('+')
    if not in_list:
        return x
    else:
        return '+'.join(sorted(x.split('+')))
    
def remove_classes_only_1_dataset(plot_dict, occ_in_dataset):
    data_aux = plot_dict.copy()
    only_1 = occ_in_dataset[occ_in_dataset['size'] == 1]['sum']
    for key in data_dict:
        df = data_aux[key].copy()
        for i in only_1.index.tolist():
            df = df[df.instrument != i]
        data_aux[key] = df
    return data_aux

def get_classes_only_in_list_of_datasets(plot_dict, datasets):
    data_aux = plot_dict.copy()
    inst_list = []
    for name in datasets:
        inst_list = inst_list + np.unique(data_aux[name]['instrument'].to_numpy()).tolist()
    inst_set = set(inst_list)
    inst_list = list(inst_set)
    for name in data_aux:
        data_aux[name] = data_aux[name][data_aux[name].instrument.isin(inst_list)]
    return data_aux

def get_MusicDelta_urls(folder):
    """
    Returns a sorted list of the files that starts with MusicDelta in the given directory
    """
    f = open(folder, 'r')
    urls = []
    line = f.readline()
    while line:
        urls.append(line.strip())
        line = f.readline()

    urls.sort()
    return urls


def get_Annotations_list_from_url(url, legend, fs):
    """
    Input: annotation .txt url 
    Output: list of the annotations, checking if 2 or more have the same timestep and joining it
            dict with the value assigned to each instrument key
    """
    reader = urlopen(url)
    values = reader.readline().split()
    annotations = [[float(values[0]),values[1].decode("utf-8"), round(fs*float(values[0]))]]
    line = reader.readline()
    while line:
        values = line.split()
        names = []
        # less than 60 ms difference is considered  same note/instrument
        if ( float(values[0]) - annotations[-1][0]) < 0.06 :
            if type(annotations[-1][1]) == str:
                names.append(annotations[-1][1] + '+' + values[1].decode("utf-8"))
                names.append(values[1].decode("utf-8") + '+' + annotations[-1][1])
            else:
                names.append(annotations[-1][1].decode("utf-8") + '+' + values[1].decode("utf-8"))
                names.append(values[1].decode("utf-8") + '+' + annotations[-1][1].decode("utf-8"))
            for i in [0,1]:
                s = names[i]
                if s in legend.keys():
                    annotations[-1][1] = s
                    break
                elif names[(i+1)%2] in legend.keys():
                    annotations[-1][1] = names[(i+1)%2]
                    break
                else:
                    annotations[-1][1] = s
                    legend[s] = len(legend) + 1
            else:
                annotations[-1][1] = s
        else:
            annotations.append([float(values[0]),values[1].decode("utf-8"), round(fs*float(values[0]))])
            if annotations[-1][1] not in legend.keys():
                legend[annotations[-1][1]] = len(legend) + 1
        line = reader.readline()
    
    reader.close()
    return annotations


def get_Annotations_list_from_file(filename, legend, fs):
    """
    Input: annotation .txt filename
    Output: list of the annotations, checking if 2 or more have the same timestep and joining it
            dict with the value assigned to each instrument key
    """
    reader = open(filename)
    val = reader.readline().split()
    values = [x.replace('e', '') for x in val]
    annotations = [[float(values[0]),values[1], round(fs*float(values[0]))]]
    line = reader.readline()
    while line:
        val = line.split()
        values = [x.replace('e', '') for x in val]
        names = []
        # less than 60 ms difference is considered  same note/instrument
        if ( float(values[0]) - annotations[-1][0]) < 0.06 :
            if type(annotations[-1][1]) == str:
                names.append(annotations[-1][1] + '+' + values[1])
                names.append(values[1] + '+' + annotations[-1][1])
            else:
                names.append(annotations[-1][1] + '+' + values[1])
                names.append(values[1] + '+' + annotations[-1][1])
            for i in [0,1]:
                s = names[i]
                if s in legend.keys():
                    annotations[-1][1] = s
                    break
                elif names[(i+1)%2] in legend.keys():
                    annotations[-1][1] = names[(i+1)%2]
                    break
                else:
                    annotations[-1][1] = s
                    legend[s] = len(legend) + 1
            else:
                annotations[-1][1] = s
        else:
            annotations.append([float(values[0]),values[1], round(fs*float(values[0]))])
            if annotations[-1][1] not in legend.keys():
                legend[annotations[-1][1]] = len(legend) + 1
        line = reader.readline()
    
    reader.close()
    return annotations


def sliceDrums_from_annotations(song_name, segments_dir, song_dict, fs):
    """
        Input:  song_name: str woth a key in the song_dict
                segments_dir : str with path where slices are saved
                song_dict : dict containing audio stream and annotations
                fs :  sampling rate to properly safe the files

        This function slices audio stream based on annotations and save each slice in a individual wav file, 
        each on the corresponent folder = segmens_dir/song_name/instrument/file.wav
        

        This function could be combined with the feature extraction in the next cells, but having the slices
        saved allows us to do data augmentation combining individual samples to get more instances of all the combinations
    """
    song = song_dict[song_name]

    od_complex = OnsetDetection(method = 'complex')
    w = Windowing(type = 'hann')
    fft = FFT() # this gives us a complex FFT
    c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)
    onsets = Onsets()

    x = song['audio']
    duration = float(len(x)) / fs

    x = x / np.max(np.abs(x))
    
    t = np.arange(len(x)) / float(fs)

    #Essentia beat tracking
    pool = Pool()
    for frame in FrameGenerator(x, frameSize = 1024, hopSize = 512):
        mag, phase, = c2p(fft(w(frame)))
        pool.add('features.complex', od_complex(mag, phase))


    onsets_list = onsets(array([pool['features.complex']]), [1])
    first_onset = int(onsets_list[0]*fs)
    
    print(first_onset)
    if not os.path.exists(segments_dir):#creating the directory
        os.mkdir(segments_dir)
    segments_dir = os.path.join(segments_dir, song_name)
    if not os.path.exists(segments_dir):#creating the directory
        os.mkdir(segments_dir)   
    n_notes = len(song['annotations'])
    annotations = song['annotations']
    file_count = 0
    for i in range(1,n_notes):
        if i != n_notes-1 and i != 0:
            x_seg = song['audio'][(annotations[i][2]-500 + first_onset):(annotations[i+1][2]-500 + first_onset)] 
            
        if len(x_seg) < 5000 or np.max(np.abs(x_seg)) < 0.05:
            continue
            
        x_seg = x_seg / np.max(np.abs(x_seg))
        
        
        x_seg_dir = os.path.join(segments_dir,annotations[i][1])
        if not os.path.exists(x_seg_dir):#creating the directory
            os.mkdir(x_seg_dir)
        path, dirs, files = next(os.walk(x_seg_dir))
        dir_n_files = len(files)
        filename = os.path.join(x_seg_dir, annotations[i][1] + '_' + str(dir_n_files) + '.wav')
        ess.MonoWriter(filename = filename, format = 'wav', sampleRate = fs)(x_seg)
        file_count = file_count + 1
        
    print(song_name + ": " + str(file_count))


    
def get_files_in_dir(dir_name):
    '''Get all path + filenames in the dir_name directory, returns an alphabetical sorted list
    Avoid .DS_Store and notebook checkpoints
    '''
    file_names = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file == ".DS_Store" or "ipynb_checkpoints" in root:
                continue
            file_name = os.path.join(root,file)
            file_names.append(file_name)
    file_names.sort()
    return file_names

def get_wavs_in_dir(dir_name):
    '''Get all path + wav in the dir_name directory, returns an alphabetical sorted list
    Avoid .DS_Store and notebook checkpoints
    '''
    file_names = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file == ".DS_Store" or "ipynb_checkpoints" in root or ".wav" not in file:
                continue
            file_name = os.path.join(root,file)
            file_names.append(file_name)
    file_names.sort()
    return file_names

def MusicDelta_filenameList_to_urlList(filename, list_, repo_url ):
    """
    Specific function to convert filenames to the corresponent url in the CarlSouthall repo
    The resulting file is used on DatasetProcessing notebook to download data
    Input : list of filenames
    Output : write a file with all the urls
    
    """
    f = open(filename, 'w')
    for line in list_:
        items = line.split()
        line = items[0] + '%20' + items[1]
        f.write(repo_url + line + '\n')
    f.close()
    
    
