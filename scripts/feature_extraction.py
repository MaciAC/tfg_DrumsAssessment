import essentia.standard as ess
from scripts.utils import get_files_in_dir
import os

def extract_MusicalFeatures(folder, descriptors, filename):
    file_count = 0
    segment_files = get_files_in_dir(folder)
    data_file = os.path.join(folder, filename)

    with open(data_file, 'w') as writer:
        
        #adding column names as the first line in csv
        line2write = ','.join(descriptors + ['instrument']).replace('lowlevel.','') + '\n'
        writer.write(line2write)
        for file in segment_files:
            if '.wav' in file:
                file_count +=1
                if file_count % 20 == 0:#print name of a file every 20 files
                    print(file_count, "files processed, current file: ",file)
                features, features_frames = ess.MusicExtractor(lowlevelSilentFrames='drop',
                                                              lowlevelFrameSize = 2048,
                                                              lowlevelHopSize = 1024,
                                                              lowlevelStats = ['mean', 'stdev'])(file)
                selected_features = [features[descriptor] for descriptor in descriptors]
                instrument = file.split('/')[-1].split('_')[0].lower()     #class information
                line2write = str(selected_features)[1:-1] + ',' + instrument + '\n'
                writer.write(line2write)
    print("A total of ",file_count, "files processed")

def get_lowLevelDescriptors():
    file = 'data/test.wav'
    features, features_frames = ess.MusicExtractor(lowlevelSilentFrames='drop',
                                                  lowlevelFrameSize = 2048,
                                                  lowlevelHopSize = 1024,
                                                  lowlevelStats = ['mean', 'stdev'])(file)

    descriptors = [descriptor for descriptor in features.descriptorNames() if 'lowlevel' in descriptor and isinstance(features[descriptor], float)]
    
    return descriptors