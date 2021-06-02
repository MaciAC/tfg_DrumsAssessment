import essentia.standard as ess
from utils import get_files_in_dir
import os
import numpy as np

def extract_MusicalFeatures(folder, descriptors, filename):
    file_count = 0
    segment_files = get_files_in_dir(folder)
    print(segment_files)
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
                instrument = file.split('/')[-1].split('_')[1].lower()[:-4]     #class information
                line2write = str(selected_features)[1:-1] + ',' + instrument + '\n'
                writer.write(line2write)
    print("A total of ",file_count, "files processed")

def get_lowLevelDescriptors(filename):
    features, features_frames = ess.MusicExtractor(lowlevelSilentFrames='drop',
                                                  lowlevelFrameSize = 2048,
                                                  lowlevelHopSize = 1024,
                                                  lowlevelStats = ['mean', 'stdev'])(filename)

    descriptors = [descriptor for descriptor in features.descriptorNames() if 'lowlevel' in descriptor and isinstance(features[descriptor], float)]
    
    return descriptors


def deviation_statistics(prefix, deviations, histogram_max):
    features = {}
    deviations = np.array(deviations)
    d2 = deviations * deviations
    variance0 = np.mean(d2)
    d4 = d2 * d2
    kurtosis0 = np.mean(d4) / (variance0 * variance0)
    std_sixth_moment0 = np.mean(d4 * d2) / (variance0 * variance0 * variance0)
    features[prefix + 'variance0'] = variance0
    features[prefix + 'kurtosis0'] = kurtosis0
    features[prefix + 'std_6th_moment0'] = std_sixth_moment0
    # histogram. Just the adoptation of Sunkalp's code.
    if (deviations.size > 0):
        point20 = 0.2 * histogram_max
        point50 = 0.5 * histogram_max
        point100 = histogram_max
        features[prefix + 'diff_0_20'] = np.where(deviations < point20)[0].size / float(deviations.size)
        features[prefix + 'diff_20_50'] = np.where((deviations < point50) & (deviations >= point20))[0].size / float(deviations.size)
        features[prefix + 'diff_50_100'] = np.where((deviations < point100) & (deviations >= point50))[0].size / float(deviations.size)
        features[prefix + 'diff_100_Inf'] = np.where(deviations >= point100)[0].size / float(deviations.size)
    else:
        features[prefix + 'diff_0_20'] = math.nan
        features[prefix + 'diff_20_50'] = math.nan
        features[prefix + 'diff_50_100'] = math.nan
        features[prefix + 'diff_100_Inf'] = math.nan

    # extremes
    extreme_margin = 0.25  # considering only top 25 %
    percentile = (1 - extreme_margin)
    ext_diff = np.sort(deviations)[int(percentile * deviations.size):]
    features[prefix + 'mean_diff_ext'] = np.mean(ext_diff)
    if len(deviations) > 0:
        features[prefix + 'diff_ext'] = np.percentile(deviations, 100 * percentile)
    else:
        features[prefix + 'diff_ext'] = 0
    return features

def timing_statistics(devs, threshold = 0.05, histogram_max = 0.3):
    features = {}
    abs_devs = np.abs(devs)
    thresholded = abs_devs - threshold
    thresholded[thresholded < 0] = 0
    features["thresholded"] = thresholded.tolist()
    features.update(deviation_statistics("dev.", abs_devs, histogram_max))
    features.update(deviation_statistics("thresholded.", thresholded, histogram_max))
    return features

def attack_deviations(reference_events, actual_onsets, start, end):
    """
    Calculates deviations of actual onsets from reference ones.
    Only the segment between 'start' and 'end' is processed.

    :param reference_events: reference event times (in seconds)
    :param actual_onsets:  actual onsts times
    :param start: start time
    :param end: end time
    :return: deviations array
    """
    filtered_actual = actual_onsets[(actual_onsets >= start) & (actual_onsets < end)]
    devs = []
    for x in filtered_actual:
        i = np.searchsorted(reference_events, x)
        if i >= len(reference_events) or (0 < i and x - reference_events[i - 1] < reference_events[i] - x):
            attack_dev = x - reference_events[i - 1]
        else:
            attack_dev = x - reference_events[i]
        devs.append(attack_dev)
    return np.array(devs)