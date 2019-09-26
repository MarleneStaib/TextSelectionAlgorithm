#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:01:19 2018

TEXT SELECTION ALGORITHM SCRIPT

@author: marlene
"""
import os
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


# keep track of the sentences selected, to avoid selecting duplicates
# (in case score drops to 0)
selected_sentence_ids = set()
selected_zeros = 0

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


##################
# CORE FUNCTIONS #
##################

# turn every sentence into a binary numpy array, indicating the base types present
def prepare_sentences(phones_dir, base_type, wishlist_ids):
    """
    pass over the corpus and turn every sentence into a binary array of length wishlist.
    base_type: phone, diphone or triphone, type: str
    wishlist_ids: np.array containing all the base types, type: str
    corpus_size: number of (input) sentences to be used for selection
    stress: whether stress is taken into account or not
    returns: array of arrays, i.e. dataset
    """
    corpus = os.listdir(phones_dir)
    # initialize an empty np.array to store the sentences
    # shape is defined by the number of sentences and the length of the wishlist
    # the latter gives the dimensionality to each sentence array
    sentences = np.zeros((len(corpus), wishlist_ids.shape[0]))
    i = 0 # keep a counter, to index into the np.array above
    for f in corpus:
        phonlist = open(os.path.join(phones_dir, f)).read().strip().split()

        if base_type == "diphones":
            units = ["{}_{}".format(phonlist[i], phonlist[i+1]) for i in
                     range(len(phonlist)-1)]
                        
        if base_type == "triphones":
            # add extra padding
            pad_phones = phonlist + ["</s>"]
            units = ["{}^{}+{}".format(pad_phones[i], pad_phones[i+1],
                                       pad_phones[i+2])
                     for i in range(len(pad_phones)-2)]
                
        # binarize them
        for unit in units:
            idx = np.where(wishlist_ids == unit)[0][0]
            sentences[i][idx] = 1
        i += 1
            
    return sentences, corpus


# scoreing function: dot each sentence with the wishlist
def score_dataset(dataset, wishlist):
    """
    takes in a binarized data set array of sentences, where every sentence is an array).
    returns the index of the highest scoring sentence. in case of tie, the first
    sentence in the list is selected.
    """
    global selected_zeros
    scores = []
    for sentence in dataset:
        # do the dot product; append to scores
        scores.append(np.dot(sentence, wishlist))
    i = 0  # use the first match in case of tie; iterate to avoid duplication in case
    # they all drop to 0
    while True:
        best_sentence_id = np.where(scores == np.max(scores))[0][i]
        if best_sentence_id in selected_sentence_ids:
            i += 1  # go to the next sentence
        else:
            if np.max(scores) == 0:
                selected_zeros += 1
            selected_sentence_ids.add(best_sentence_id)
            return best_sentence_id


# main algorithm
def text_selection(in_dir, out_dir, base_type, scoring_type):
    """
    greedy selection of highest-scoring sentences, according to a wishlist.
    wishlist is a numpy array of either all 1s or weights/probabilities (for each
    base type). num_sentences is the number to be selected. writes the selected
    sentences to a file and returns the distribtuion of base types in the selected 
    corpus (for plotting).
    base type: "phones", "diphones", "triphones"
    stress: consider stress or not
    scoring_type: 1 for vanilla, scoring each unit equally, 2 for weighted scores
    num_sentences: the number of sentences selected
    """
    # load the wishlist and unit ids
    name = base_type
    wishlist = np.load("{}_wishlist{}.npy".format(name, scoring_type))
    ids = np.load("{}_ids.npy".format(name))
    corpus_t = 0  # the total number of seconds in the corpus so far

    # load the data set
    phones_dir = os.path.join(in_dir, 'phonemes')
    dataset, filenames = prepare_sentences(phones_dir, base_type, ids)
    
    # while corpus_t < 3600:
    while np.where(wishlist == 0)[0].shape[0] < wishlist.shape[0]:
        best_sentence_id = score_dataset(dataset, wishlist)
        # select the sentence from the original dataset and write to a file
        sourcedir = os.path.join(in_dir, 'phonemes', filenames[best_sentence_id])
        targetdir = os.path.join(out_dir, 'phonemes', filenames[best_sentence_id])
        shutil.copyfile(sourcedir, targetdir)

        # also copy the mel
        sourcedir = os.path.join(in_dir, 'mel',
                                 filenames[best_sentence_id].replace('.txt', '.npy'))
        targetdir = os.path.join(out_dir, 'mel',
                                 filenames[best_sentence_id].replace('.txt', '.npy'))
        shutil.copyfile(sourcedir, targetdir)

        # asses the length of wav
        wavfile = os.path.join(in_dir, 'wav_24khz',
                               filenames[best_sentence_id].replace('.txt', '.wav'))
        sr, w = read(wavfile)
        l = len(w) / sr
        # add that to the total length of selected corpus
        corpus_t += l

        best_sentence_arr = dataset[best_sentence_id]
        # delete the selected sentence from the wishlist
        wishlist[np.where(best_sentence_arr != 0)] = 0
        # add the new sentence to the distribution
        # distribution = distribution + best_sentence_arr
        
    # return distribution
    print("number of units selected: {} out of {}".format(
        np.where(wishlist == 0)[0].shape[0],
          wishlist.shape[0]))
    print("number of selected hours: {}".format(corpus_t / 3600))


# plot the resulting distributions
def plot_dist(dist_array, name, plotname):
    f = plt.figure()
    plt.bar(range(dist_array.shape[0]), dist_array.sort_values(ascending=False))
    plt.xlabel("Rank", size=17)
    plt.ylabel("Frequency", size=17)
    plt.title("Distribution of {}".format(plotname), size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    f.savefig("{}.pdf".format(name), format="pdf")



########
# MAIN #
########


def main():
    # baseline: randomly sample 5XX sentences, measure coverage
    # call text_selection for different base types/wishlists
    # wishlist1 gives equal weight to all new base units
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Phonemes directory')
    parser.add_argument('--out-dir', type=str, required=True, help='Target directory')
    parser.add_argument('--unit-type', type=str, default='diphones',
                        help='diphones/triphones')
    parser.add_argument('--scoring-type', type=int, default=2,
                        help='choose 2 for negative log probabilities')
    ARGS = parser.parse_args()
    os.makedirs(os.path.join(ARGS.out_dir, 'phonemes'), exist_ok=True)
    os.makedirs(os.path.join(ARGS.out_dir, 'mel'), exist_ok=True)

    text_selection(ARGS.data, ARGS.out_dir, ARGS.unit_type, scoring_type=ARGS.scoring_type)


if __name__ == "__main__":
    main()
    print("selected zeros:", selected_zeros)
