from scikits.talkbox.features import mfcc
from sklearn import hmm

from scipy.io import wavfile
import numpy as np
import os
import pickle
from dtw import Dtw


class SpeechRecognizer:
    def __init__(self):
        self.mfcc_dictionary = {}
        self.hmm_dictionary = {}
        self.sound = None
        self.sound_samplerate = None

    def get_sound_mfcc(self, input):
        return mfcc(input, nwin=int(self.sound_samplerate * 0.03), fs=self.sound_samplerate, nceps=13)[0]

    @staticmethod
    def get_sound_hmm(word_mfcc):
        model = hmm.GaussianHMM(15, "full")
        model.fit([word_mfcc])
        return model

    def load_sound(self, filename):
        self.sound_samplerate, self.sound = wavfile.read(filename)

    def save_dictionary(self):
        with open('mfcc.db', 'wb') as f:
            pickle.dump(self.mfcc_dictionary, f)
        with open("hmm.db", 'wb') as f:
            pickle.dump(self.hmm_dictionary, f)

    def load_dictionary(self):
        with open('mfcc.db', 'rb') as f:
            self.mfcc_dictionary = pickle.load(f)
        with open("hmm.db", 'rb') as f:
            self.hmm_dictionary = pickle.load(f)

    @staticmethod
    def distance(a, b):
        result = (a-b)**2
        result = result.sum(axis=-1)
        result = np.sqrt(result)
        return result

    def compare(self, a, b):
        #if len(a) > len(b):
        #    a = a[:len(b)]
        #if len(b) > len(a):
        #    b = b[:len(a)]
        dtw = Dtw(a, b, distance_func=self.distance)
        result = dtw.calculate()
        return result

    def search_word_dtw(self, word_mfcc):
        distances = {}
        for k, v in self.mfcc_dictionary.iteritems():
            distances[k] = self.compare(word_mfcc, v)
        return min(distances, key=distances.get)

    def search_word_hmm(self, word_mfcc):
        scores = {}
        for k, v in self.hmm_dictionary.iteritems():
            scores[k] = v.score(word_mfcc)
        return max(scores, key=scores.get)

    def load_wavs_from_dir(self, dir_name):
        wav_file_list = [f for f in os.listdir(dir_name) if os.path.splitext(f)[1] == '.wav']
        for wav_file in wav_file_list:
            self.load_sound(dir_name + '/' + wav_file)
            mfcc_temp = self.get_sound_mfcc(self.sound)
            self.mfcc_dictionary[os.path.splitext(wav_file)[0]] = mfcc_temp
            self.hmm_dictionary[os.path.splitext(wav_file)[0]] = self.get_sound_hmm(mfcc_temp)

    def search_wav_hmm(self, filename):
        self.load_sound(filename)
        return self.search_word_hmm(self.get_sound_mfcc(self.sound))

    def search_wav_dtw(self, filename):
        self.load_sound(filename)
        return self.search_word_dtw(self.get_sound_mfcc(self.sound))


def main():
    sr = SpeechRecognizer()
    sr.load_wavs_from_dir('learningwavs')
    sr.save_dictionary()
    sr.load_dictionary()
    #print sr.hmm_dictionary

    print "DTW Test:"
    print sr.search_wav_dtw("testingwavs/1_1.wav")
    print sr.search_wav_dtw("testingwavs/1_2.wav")
    print sr.search_wav_dtw("testingwavs/1_3.wav")
    print sr.search_wav_dtw("testingwavs/1_4.wav")
    print sr.search_wav_dtw("testingwavs/1_5.wav")
    print sr.search_wav_dtw("testingwavs/1_6.wav")
    print sr.search_wav_dtw("testingwavs/1_7.wav")
    print sr.search_wav_dtw("testingwavs/1_8.wav")
    print sr.search_wav_dtw("testingwavs/1_9.wav")
    print sr.search_wav_dtw("testingwavs/1_10.wav")

    print("\nHMM Test")
    print sr.search_wav_hmm("testingwavs/1_1.wav")
    print sr.search_wav_hmm("testingwavs/1_2.wav")
    print sr.search_wav_hmm("testingwavs/1_3.wav")
    print sr.search_wav_hmm("testingwavs/1_4.wav")
    print sr.search_wav_hmm("testingwavs/1_5.wav")
    print sr.search_wav_hmm("testingwavs/1_6.wav")
    print sr.search_wav_hmm("testingwavs/1_7.wav")
    print sr.search_wav_hmm("testingwavs/1_8.wav")
    print sr.search_wav_hmm("testingwavs/1_9.wav")
    print sr.search_wav_hmm("testingwavs/1_10.wav")

if __name__ == "__main__":
    main()
