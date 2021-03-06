#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:01:53 2018

EXTRACT DISTRIBUTIONS AND WISHLIST FOR TEXT SELECTION ALGORITHM

@author: marlene
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import timeit

#################
# CONFIGURATION #
#################

DEBUG = 1000
ARCTIC = 60000
FULL = 581905
num_sentences = DEBUG #set to DEBUG, ARCTIC or FULL
utt_file = "example_utts.txt" #the input file of utterance data

##################
# CORE FUNCTIONS #
##################

def collect_stats(utt_file, num_sentences):
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

    phones = {}
    diphones = {}
    triphones = {}
    phones_stress = {}
    diphones_stress = {}
    triphones_stress = {}
    
    with open(utt_file, "r") as corpus:
        for line in corpus.readlines()[:num_sentences]:
            if line.startswith("utt"):
                #with open(utts_clean, "a") as outfile:
                #    outfile.write(line)
                #strip the utterance number away
                line = re.sub("utt[0-9]+", "", line)
                #replace the break symbols by silence; here I just assume silence for 
                #either a small (B) or big break (BB)
                line = re.sub("_BB?", "(0 sil )", line)
                #words = line.split("}{")
                syllables = re.findall("\(([0-9] (?:[a-z@!\^]+ *)+)\)", line)
                #pad the phones with an extra 'sil' in the beginning
                phonlist = ["sil"] + re.findall("[a-z@!\^]+", line)
                stressed_phones = ["sil"]
                
                for syl in syllables:
                    #find the stress for that syl, either 0,1,2 or 3
                    syl_tmp = syl.split()
                    stress = syl_tmp[0]
                    #add the stress to every vowel, add them to the list of stressed phones
                    for p in syl_tmp[1:]:
                        if re.match("[aeiou@]", p): 
                            stressed_phones.append(stress + p)
                        else:
                            stressed_phones.append(p)
                
                #extract phone distribution
                for phon in phonlist:
                    #if phon in phones:
                    try:
                        phones[phon] += 1
                    except:
                        phones[phon] = 1
                
                #extract distribution of stressed phones
                for phon in stressed_phones:
                    try:
                        phones_stress[phon] += 1
                    except:
                        phones_stress[phon] = 1
                
                
                #extract diphones from phones
                for i in range(len(phonlist)-1):
                    dip = "{}_{}".format(phonlist[i], phonlist[i+1])
                    #if dip in diphones:
                    try:
                        diphones[dip] += 1
                    except:
                        diphones[dip] = 1
                        
                #also extract stressed diphones
                for i in range(len(stressed_phones)-1):
                    dip = "{}_{}".format(stressed_phones[i], stressed_phones[i+1])
                    try:
                        diphones_stress[dip] +=1
                    except:
                        diphones_stress[dip] = 1
                
                #extract triphones
                #add extra padding
                pad_phones = phonlist + ["sil"]
                for i in range(len(pad_phones)-2):
                    tri = "{}^{}+{}".format(pad_phones[i], pad_phones[i+1], pad_phones[i+2])
                    #if tri in triphones:
                    try:
                        triphones[tri] += 1
                    except:
                        triphones[tri] = 1
                
                #extract stressed triphones
                pad_phones = stressed_phones + ["sil"] #extra padding, use stressed phones
                for i in range(len(pad_phones)-2):
                    tri = "{}^{}+{}".format(pad_phones[i], pad_phones[i+1], pad_phones[i+2])
                    try:
                        triphones_stress[tri] += 1
                    except:
                        triphones_stress[tri] = 1
                        
    return phones, diphones, triphones, phones_stress, diphones_stress, triphones_stress


#plot the distributions
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
    
    df = df.sort_values(0) #sort by the di/tri/phone name
    ids = np.array(df[0])
    wishlist1 = np.full(df[0].shape, 1) #the basic wishlist just has 1s for every base type
    #the second wishlist is based on type frequency
    wishlist2 = np.array(df[1])
    wishlist2 = -np.log(wishlist2/np.sum(wishlist2)) #turn them into neg. log probs
    #save them as np arrays for later use
    np.save("{}_ids.npy".format(name), ids)
    np.save("{}_wishlist1.npy".format(name), wishlist1)
    np.save("{}_wishlist2.npy".format(name), wishlist2)


########
# MAIN #
########

def main():
    
    #get the stats from the data
    inventories = collect_stats(utt_file, num_sentences)
    
    #names for saving and plotting
    names = ["phones", "diphones", "triphones", "phones_stress", "diphones_stress", "triphones_stress"]
    plotnames = ["phones", "diphones", "triphones", "phones incl. stress", "diphones incl. stress", "triphones incl. stress"]
    
    print(sum(inventories[0].values()))
    
    #plot their distributions, save the wishlists
    for i, inv in enumerate(inventories):
        df = pd.DataFrame(list(inv.items()))
        plot_dist(df, names[i], plotnames[i])
        save_wishlists(df, names[i])
    

if __name__ == "__main__":
    main()
    #print(timeit.timeit('main()', setup="from __main__ import main", number=1))