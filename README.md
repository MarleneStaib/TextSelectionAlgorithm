# TextSelectionAlgorithm

This is just some example code I wrote for a class project. It could be more nicely modularized, which I haven't gotten to yet, but I find the basic idea quite simple and nice.

It's a flexible implementation of an algorithm for phonetically rich text selection, as used for designing prompt scripts for unit selection speech synthesis. From a large input corpus, this greedily, iteratively selects the sentence with the most novel phonetic material, and returns a (pre-specified) number of sentences.

The basic principle is to select sentences with previously unseen phonological material in them, in order to pick up as many different units (e.g., diphones) as possible. This implementation allows to flexibly choose among 6 different unit types: phones/diphones/triphones, each with or without stress annotation. It also allows for two types of scoring method: 

1. Each sentence receives a score of 1 for every not previously selected unit 
2. Each sentence receives a score for every not previously selected unit, which is inverse-proportional to the unit's frequency in the input corpus (using -log(frequency) )

Scores for every sentence are recomputed after every iteration (selecting 1 sentence at a time).
The idea behind this implementation is to speed up these computations by using numpy array and matrix operations.
Every unit type/scoring method has a corresponding 'wishlist', which is a numpy array that stores either 1 or -log(frequency) for every base unit type. 

preprocess.py extracts these counts from a large corpus of input data, saves them as numpy arrays and plots the distributions of phones/diphones/triphones in the data.

example_utts.txt shows the format of the input data. This (phonetic) representation of the original (text) corpus is obtained by running text through the festival TTS frontend.

example_text.txt shows that text in its normal, graphemic representation. 

text_selection_algorithm.py uses the extracted 'wishlists' in its scoring function. For each type of optimization (phone/diphone/triphone, +/- stress, uniform versus inverse-frequency based) there is a different wishlist, and the configuration determines which wishlist is beging used. 

First, the input corpus is turned into a 2-D array of dimensionality (# of sentences) x (# of units in the selected wishlist). For every sentence, this matrix indicates whether a given unit is contained in the respective sentence. Scores can then be easily obtained by taking the vector-matrix product of the wishlist (w) with the corpus matrix (C): 
                                                    w C
                                                    
The maximum scoring sentence is selected, and the entries for the found units are set to 0 in the wishlist.
On the upside, this means that C only gets constructed once, and never has to be updated (scores for previously extracted sentences will automatically turn 0). Only small updates to w are necessary, making this a time-efficient alternative to looping over the input corpus (again and again).
On the downside, this implementation is not particularly space-efficient, and, e.g., running triphone-based optimization with a corpus of around 100k sentences requires additional memory.
