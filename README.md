# Interpretable-contrastive-word-mover-s-embedding
# Paper Datasets
Here is a Dropbox link to the datasets used in the paper: https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
The dataset in the above link was provided in .mat file. You may need to transform to the .npy file to run our code.
Each mat file contains following component<br />
**X** is a cell array of all documents, each represented by a dxm matrix where d is the dimensionality of the word embedding and m is the number of unique words in the document. ("BBCsports.npy")<br />
**Y** is an array of labels ("BBCsports_grade.npy")<br />
**BOW_X** is a cell array of word counts for each document('weight.npy')<br />
**indices** is a cell array of global unique IDs for words in a document<br />
**TR** is a matrix whose ith row is the ith training split of document indices('index_tr.npy')<br />
**TE** is a matrix whose ith row is the ith testing split of document indices('index_te.npy')<br />
'BBCsports_length.npy' is the number of unique words for each sample.
# Demo
In the demo code we use BBCsports data set. The data is preprocessed and has been saved as .npy file can be found in the following link:
https://drive.google.com/drive/folders/1GuQsHS1J8J24GnCmTCTDPH5hWWYtmw4s?usp=sharing <br />
Please put the data into the same path as 2 python files.<br />
Use 
```
python run_pos.py
```
to run the file.
# Citation
If you find this repo useful for your research, please consider citing the paper<br />
```
@misc{jiang2021interpretable,
    title={Interpretable contrastive word mover's embedding},
    author={Ruijie Jiang and Julia Gouvea and Eric Miller and David Hammer and Shuchin Aeron},
    year={2021},
    eprint={2111.01023},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
Any question please feel free to contact Ruijie Jiang (Ruijie.Jiang@tufts.edu).

