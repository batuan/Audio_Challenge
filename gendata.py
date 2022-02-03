import json
import os
import librosa
import numpy as np

dir = './vad_data/'
files = os.listdir(dir)
json_files = [f for f in files if f.endswith('json')]
SAMPLE_RATE = 16000
TRACK_DURATION = 0.064
SAMPLES_PER_TRACK = int(TRACK_DURATION * SAMPLE_RATE)


def extract_features(signal, sr=16000, n_mfcc=5, size=512, step=16, n_mels=40):
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


def get_data(json_file, segments, seg_len=SAMPLES_PER_TRACK):
    audio_file = dir + json_file.split('.')[0] + '.wav'
    print(audio_file)
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    audio_len = len(audio)
    start = 0
    end = seg_len
    seg_data = []
    seg_label = []

    for seg in segments:
        begin_seg, end_seg = int(seg[0] * SAMPLE_RATE), int(seg[1] * SAMPLE_RATE)
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

data = []
labels = []

for file in json_files:
    path = dir + file
    segments = []
    with open(path) as f:
        d = json.load(f)
        dict_segments = d['speech_segments']
        for seg in dict_segments:
            segments.append([round(float(seg['start_time']), 2), round(float(seg['end_time']), 2)])

    seg_data, seg_label = get_data(file, segments)
    data.extend(seg_data)
    labels.extend(seg_label)

# features = [extract_features(d) for d in data]
# print(np.asarray(features).shape)

import multiprocessing
# p = multiprocessing.Pool(6)

# train_data = [extract_features(d) for d in data] #p.map(extract_features, data)
# np.save('train_imgs.npy', train_data)
np.save('train_labels.npy', labels)
# print(labels)
