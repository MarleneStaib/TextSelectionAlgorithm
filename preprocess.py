#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:01:53 2018

EXTRACT DISTRIBUTIONS AND WISHLIST FOR TEXT SELECTION ALGORITHM

@author: marlene
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

##################
# CORE FUNCTIONS #
##################


def collect_stats(phones_dir):
    """
    Extract the counts of phones/diphones/triphones, with or without stress, from
    the input corpus.
    
    type utt_file: readable file
    param utt_file: input corpus (used for selection)
    
    type num_sentences: int
    param num_sentences: number of sentences to be used from the corpus; optionally
                         use to truncate the corpus (to speed up or debug text
                         selection, to make compatible with the ARCTIC Text Selection)
    
    return type: list of dictionaries
    returns: a dictionary for each type of base unit, with the ids as key and the 
             counts of those units as values
    """

    monophones = {}
    diphones = {}
    triphones = {}
    
    for fname in os.listdir(phones_dir):
        phones = open(os.path.join(phones_dir, fname)).read().strip().split()
        # extract phone distribution
        for phon in phones:
            if phon not in monophones:
                monophones[phon] = 0
            monophones[phon] += 1

            # extract diphones from phones
            for i in range(len(phones) - 1):
                dip = "{}_{}".format(phones[i], phones[i+1])
                if dip not in diphones:
                    diphones[dip] = 0
                diphones[dip] += 1
                
            # extract triphones
            # add extra padding
            pad_phones = phones + ["</s>"]
            for i in range(len(pad_phones)-2):
                tri = "{}^{}+{}".format(pad_phones[i], pad_phones[i+1], pad_phones[i+2])
                if tri not in triphones:
                    triphones[tri] = 0
                triphones[tri] += 1
    return monophones, diphones, triphones


# plot the distributions
def plot_dist(dist_df, name, plotname):
    """Function that plots the distribution of phones/diphones/triphones in the 
    input data. 
    
    type dist_df: pandas.DataFrame()
    param dist_df: data frame with counts of all phones/diphones/triphones in the
                   input data
    type name: str
    param name: name of the plot in saved output (pdf)
    type plotname: str
    param plotname: name in the title of the plot; choose "phone/diphone/triphone"
    """
    
    f = plt.figure()
    plt.bar(range(dist_df[0].shape[0]), dist_df[1].sort_values(ascending=False))
    plt.xlabel("Rank", size=14)
    plt.ylabel("Frequency", size=14)
    plt.title("Distribution of {}".format(plotname), size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    f.savefig("{}.pdf".format(name), format="pdf")


#get wishlist
def save_wishlists(df, name):
    """Save the 'wishlists' (id array and count/all-1-array) as numpy array for 
    later reuse in text_selection_algorithm.py
    
    type dist_df: pandas.DataFrame()
    param dist_df: data frame with counts of all phones/diphones/triphones in the
                   input data
    type name: str
    param name: name of wishlist/output file
    """
    
    df = df.sort_values(0) # sort by the di/tri/phone name
    ids = np.array(df[0])
    wishlist1 = np.full(df[0].shape, 1) # the basic wishlist just has 1s for every base type
    # the second wishlist is based on type frequency
    wishlist2 = np.array(df[1])
    wishlist2 = -np.log(wishlist2/np.sum(wishlist2)) # turn them into neg. log probs
    # save them as np arrays for later use
    np.save("{}_ids.npy".format(name), ids)
    np.save("{}_wishlist1.npy".format(name), wishlist1)
    np.save("{}_wishlist2.npy".format(name), wishlist2)


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Phonemes directory')
    ARGS = parser.parse_args()
    
    # get the stats from the data
    inventories = collect_stats(ARGS.data)
    
    # names for saving and plotting
    names = ["phones", "diphones", "triphones"]
    plotnames = ["phones", "diphones", "triphones"]
    
    print(sum(inventories[0].values()))
    print("diphones: ", len(inventories[1].keys()))
    print("triphones: ", len(inventories[2].keys()))
    
    # plot their distributions, save the wishlists
    # for i, inv in enumerate(inventories):
    #     df = pd.DataFrame(list(inv.items()))
    #     plot_dist(df, names[i], plotnames[i])
    #     save_wishlists(df, names[i])
    

if __name__ == "__main__":
    main()
