# oro-facial_classifier
Binary Classifier of oro-facial expressions, born as a university class project by G.Angelini e M.Raimondi.
Class in Analysis of Biomedical Data and Signals, prof. Vittorio Sanguineti, University of Genoa(Italy), 2015-2016.
Based on dataset for research purpose from FEI Face Database, http://fei.edu.br/~cet/facedatabase.html

Edited by Gianfrancesco Angelini
Last-update 12-01-2017

##How to make it work##
1. You can just open main.m and click Run.
2. ...Or you can customize your analysis changing parameter in the 2 sections inside main.m General settings and Analysis cycle settings.

##How IT works (default)##
1. It downloads automatically the dataset and organize it in a folder.
2. Prepares the dataset labeling it with a "a" or a "b" in the pics names.
3. Reduces dimensions by PCA, SVD, or image-resize.
4. Loops for a set of explained-variances (for pca and svd):
    1. Trains a LDA, a NN and a Naive Bayes with the training set.
    2. Classifies the dataset excluding the test set.
5. Shows results
    1. Shows classifiers performances, with ROC curves and confusion matrix.
    2. Shows the best features for PDA/SVD and LDA.
