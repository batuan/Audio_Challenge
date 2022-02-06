import json
import multiprocessing
import os
import librosa
import numpy as np

dir = './vad_data/'
SAMPLE_RATE = 16000
TRACK_DURATION = 0.064
SAMPLES_PER_TRACK = int(TRACK_DURATION * SAMPLE_RATE)


def extract_features(signal, sr=16000, n_mfcc=5, size=512, step=16, n_mels=40):
    """
    Parameters
    ----------
    signal : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
        sampling rate of ``y``

    size : FFT window
    
    step : hop length

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return
    
    n_mels: number of Mel bands to generate

    Returns
    -------
    features : numpy array that contain all feature extract (mfcc, delta 1, 2, rmse)
    """

    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc, n_fft=size, hop_length=step)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_2 = librosa.feature.delta(mfcc, order=2)

    mel_spectogram = librosa.feature.melspectrogram(signal, sr=sr, n_mels=n_mels, n_fft=size, hop_length=step)
    rmse = librosa.feature.rms(S=mel_spectogram, frame_length=n_mels * 2 - 1, hop_length=step)

    mfcc, mfcc_delta, mfcc_delta_2, rmse = np.asarray(mfcc), np.asarray(mfcc_delta), np.asarray(mfcc_delta_2), np.asarray(rmse)
    # print(mfcc.shape, mfcc_delta.shape, mfcc_delta_2.shape, rmse.shape)
    features = np.concatenate((mfcc, mfcc_delta, mfcc_delta_2, rmse), axis=0)
    return features.transpose()


def segment_signal(signal, ):
    pass

total_len = 0
total_voice = 0


def get_data(json_file, segments, seg_len=SAMPLES_PER_TRACK):
    audio_file = dir + json_file.split('.')[0] + '.wav'
    print(audio_file)
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    audio_len = len(audio)
    global total_len
    total_len += audio_len
    start = 0
    end = seg_len
    seg_data = []
    seg_label = []

    for seg in segments:
        begin_seg, end_seg = int(seg[0] * SAMPLE_RATE), int(seg[1] * SAMPLE_RATE)
        global total_voice
        total_voice += (end_seg-begin_seg)
        while end < begin_seg:
            seg_data.append(audio[start: end])
            seg_label.append(0)

            start += seg_len
            end += seg_len

        start = begin_seg
        end = begin_seg + seg_len

        duration = end_seg - begin_seg
        nb_segment = duration // SAMPLES_PER_TRACK
        for i in range(nb_segment):
            seg_data.append(audio[start: end])
            seg_label.append(1)
            start += seg_len
            end += seg_len

        start = end_seg
        end = end_seg + seg_len

    while end < audio_len:
        seg_data.append(audio[start:end])
        seg_label.append(0)

        start += seg_len
        end += seg_len

    return seg_data, seg_label

def write_to_file(files):
    all_data = [['path', 'start', 'end', 'duration']]


def split_train_test(files, ratio=0.2, random_seed=42):
    files = np.asarray(files)
    np.random.seed(random_seed)
    np.random.shuffle(files)
    num_train = int(len(files) * (1-ratio))
    return files[:num_train], files[num_train:]


def get_time_segments(path):
    segments = []
    with open(path) as f:
        d = json.load(f)
        dict_segments = d['speech_segments']
        for seg in dict_segments:
            segments.append([round(float(seg['start_time']), 2), round(float(seg['end_time']), 2)])
    return segments


def gen_data_from_list_files(listfile):
    data = []
    labels = []
    for file in listfile:
        path = dir + file
        segments = get_time_segments(path)
        seg_data, seg_label = get_data(file, segments)
        data.extend(seg_data)
        labels.extend(seg_label)

    return data, labels


def list_to_file(list, name='train_list'):
    with open(name, 'w') as f:
        for item in list:
            f.write("%s\n" % item)

def number2str(x):
    if x < 10:
        return '0'+str(x)
    else:
        return str(x)


def gen_vad_data():
    files = os.listdir(dir)
    json_files = [f for f in files if f.endswith('json')]
    train_list, test_list = split_train_test(json_files, 0.2, 42)

    list_to_file(train_list, './train_test_list/train_list.txt')
    list_to_file(test_list, './train_test_list/test_lisst.txt')

    train_data, train_label = gen_data_from_list_files(train_list)
    test_data, test_label = gen_data_from_list_files(test_list)

    # p = multiprocessing.Pool(4)
    train_out = [extract_features(d) for d in train_data]  # p.map(extract_features, train_data)
    test_out = [extract_features(d) for d in test_data]  # p.map(extract_features, test_data)
    
    #TODO: multi thread
    # p.close()
    # p.join()

    np.save('./data_train_test/train_imgs.npy', train_out)
    np.save('./data_train_test/train_labels.npy', train_label)
    np.save('./data_train_test/test_imgs.npy', test_out)
    np.save('./data_train_test/test_labels.npy', test_label)


def trim_dat_noise(audio_file, seg_len=SAMPLES_PER_TRACK):
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    audio_len = len(audio)
    nb_segments = audio_len // seg_len
    seg_data = []
    for i in range(nb_segments):
        seg_data.append(audio[i*seg_len:(i+1)*seg_len])
    return seg_data, np.ones(nb_segments)


#TODO: processing noise data 
def gen_noise_data():
    ROOMS = ['Room0'+number2str(x) for x in range(0,20)]
    path1 = './RIRS_NOISES/pointsource_noises/'
    path2 = './RIRS_NOISES/real_rirs_isotropic_noises/'
    path3 = './RIRS_NOISES/simulated_rirs/'

    files = [f for f in os.listdir(path1) if f.endswith('.wav')]
    noise_data = []
    noise_labels = []
    for file in files:
        seg_data, seg_label = trim_dat_noise(path1+file)
        noise_data.extend(seg_data)
        noise_labels.extend(seg_label)
    print(len(noise_data), len(noise_labels))


if __name__ == "__main__":
    #gen_noise_data()
    gen_vad_data()