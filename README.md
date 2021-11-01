# Interpretable-contrastive-word-mover-s-embedding
# Paper Datasets
Here is a Dropbox link to the datasets used in the paper: https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# Demo
In the demo code we use BBCsports data set. The dataset in the above link was provided in matlab file. You may need to transform to the .npy file to run our code.
Each mat file contains<br />
**X** is a cell array of all documents, each represented by a dxm matrix where d is the dimensionality of the word embedding and m is the number of unique words in the document. ("BBCsports.npy")<br />
**Y** is an array of labels ("BBCsports_grade.npy")<br />
**BOW_X** is a cell array of word counts for each document('weight.npy')<br />
**indices** is a cell array of global unique IDs for words in a document<br />
**TR** is a matrix whose ith row is the ith training split of document indices('index_tr.npy')<br />
**TE** is a matrix whose ith row is the ith testing split of document indices('index_te.npy')<br />
'BBCsports_length.npy' is the number of unique words for each sample.
# Citation
Any question please feel free to contact Ruijie Jiang (Ruijie.Jiang@tufts.edu).
