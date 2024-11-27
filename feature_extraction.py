# Importing required libraries
# sklearn

import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import os
# import IPython.display as ipd  # To play sound in the notebook
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("./dataset/audio_dataset.csv")
df = pd.DataFrame(data)

# Path to audio files
def createPath(code):
  return "/dataset/audio/" + str(code) + '.mp3'
df['path'] = df['Code'].apply(createPath)
df.drop('Code', axis=1, inplace=True)


# Encoding the labels from quadrant to emotions
# 0 - Happy
# 1 - Sad
# 2 - Angry
# 3 - Fear
def encodeLabels():
    lb = LabelEncoder()
    d = lb.fit_transform(df['Label'])
    df['label'] = d
    df.drop('Label', axis=1, inplace=True)

encodeLabels()


# Extracting features from audio files
df_features = pd.DataFrame(columns=['feature'])
counter = 0
def appendResults(path):
  try:
    y, sr = librosa.load(path, duration=25 ,sr=22050*2 ,offset=3)
    y, _ = librosa.effects.trim(y)
    bpm = extractBPM(y, sr)
    mfccs = extractMFCC(y, sr)
    rms = extractRMS(y, sr)
    zcr = extractZCR(y, sr)
    chroma_stft = extractChromaSTFT(y, sr)
    spectral_centroid = extractSpectralCentroid(y, sr)
    spectral_bandwidth = extractSpectralBandwidth(y, sr)
    spectral_rolloff = extractSpectralRolloff(y, sr)
    spectral_contrast = extractSpectralContrast(y, sr)
    mel_spectogram = extractMelSpectogram(y, sr)
    amplitude = extractAmplitude(y, sr)
  except:
    df.drop(df[df['path'] == path].index, inplace=True)
    return None
  return [bpm, mfccs, rms, zcr, chroma_stft, spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast, mel_spectogram, amplitude]

def extractAmplitude(y, sr):
  S = np.abs(librosa.stft(y))
  amp = librosa.amplitude_to_db(S)
  print("amp: ", len(amp))
  return amp

def extractMFCC(y, sr):
  mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
  print("mfccs: ", len(mfccs))
  return mfccs

def extractRMS(y, sr):
  rms = librosa.feature.rms(y=y)
  print("rms: ", len(rms))
  return rms

def extractZCR(y, sr):
  zcr = librosa.feature.zero_crossing_rate(y)
  print("zcr: ", len(zcr))
  return zcr

def extractChromaSTFT(y, sr):
  chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
  print("chroma_stf: ", len(chroma_stft))
  return chroma_stft

def extractSpectralCentroid(y, sr):
  cent = librosa.feature.spectral_centroid(y=y, sr=sr)
  print("cent: ", len(cent))
  return cent

def extractSpectralBandwidth(y, sr):
  spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  print("spec_bw: ", len(spec_bw))
  return spec_bw

def extractSpectralRolloff(y, sr):
  rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
  print("rolloff: ", len(rolloff))
  return rolloff

def extractSpectralContrast(y, sr):
  contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
  print("contrast: ", len(contrast))
  return contrast

def extractMelSpectogram(y, sr):
  melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
  print("melspec: ", len(melspec))
  return melspec

def extractBPM(y, sr):
  tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
  onset_env = librosa.onset.onset_strength(y=y, sr=sr)
  tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
  print("tempo: ", len(tempo))
  return tempo


df[['bpm', 'mfccs', 'rms', 'zcr', 'chroma_stft', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_contrast', 'mel_spectogram', 'amplitude']] = df['path'].apply(lambda x: pd.Series(appendResults(x)))

# saving the dataframe to a csv file
df.to_csv('./dataset/extracted/audio_features_25_soxrHQ_44_all_stacked.csv', index=False)
df_verify = pd.read_csv('./dataset/extracted/audio_features_25_soxrHQ_44_all_stacked.csv')
print(df_verify)