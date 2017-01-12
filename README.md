# oro-facial_classifier
Binary Classifier of oro-facial expressions, born as a university class project by G.Angelini e M.Raimondi.
Class in Analysis of Biomedical Data and Signals, prof. Vittorio Sanguineti, University of Genoa(Italy), 2015-2016.
Based on dataset for <<research>> purpose from FEI Face Database, http://fei.edu.br/~cet/facedatabase.html

Edited by Gianfrancesco Angelini
Last-update 12-01-2017

How to make it work
0. You can just open main.m and click Run.
1. ...Or you can customize your analysis changing parameter in the 2 sections inside main.m General settings and Analysis cycle settings.

How IT works (default)
I.    It downloads automatically the dataset and organize it in a folder.
II.   Prepares the dataset labeling it with a "a" or a "b" in the name.
III.  Reduces dimensions by PCA, SVD, or image-resize.
IV.  Loops for a set of explained-variances (for pca and svd):
    IV.1   Classify through a LDA, a NN and a Naive Bayes.
V.   Shows results
           V.1    Shows perf
