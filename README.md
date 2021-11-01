# Interpretable-contrastive-word-mover-s-embedding
# Paper Datasets
Here is a Dropbox link to the datasets used in the paper: https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# Demo
In the demo code we use BBCsports data set. The dataset in the above link was provided in matlab file. 
Each mat file contains<br />
**X** is a cell array of all documents, each represented by a dxm matrix where d is the dimensionality of the word embedding and m is the number of unique words in the document
**Y** is an array of labels<br />
**BOW_X** is a cell array of word counts for each document<br />
**indices** is a cell array of global unique IDs for words in a document<br />
**TR** is a matrix whose ith row is the ith training split of document indices<br />
**TE** is a matrix whose ith row is the ith testing split of document indices<br />
