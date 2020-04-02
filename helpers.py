import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf

#test
# map a label to a string
label2str = {1: "PER", 2: "LOC", 3: "ORG", 4: "MISC", 5: "O"}

# predefine a label_set: PER - 1, LOC - 2, ORG - 3, MISC - 4, O - 5
# 0 is for padding
labels_map = {'B-ORG': 3, 'O': 5, 'B-MISC': 4, 'B-PER': 1,
              'I-PER': 1, 'B-LOC': 2, 'I-ORG': 3, 'I-MISC': 4, 'I-LOC': 2}

def word2features(sent, i):
    word = sent[i][0]
    #postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        #'postag=' + postag,
        #'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        #postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            #'-1:postag=' + postag1,
            #'-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        #postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            #'+1:postag=' + postag1,
            #'+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
        
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    # print sent
    return [label for token, label in sent]

    
def load_data2labels(input_file):
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    de = False
    if input_file.startswith("de."):
        de = True
    print('opening ', 'data/' + input_file)
    with open('data/' + input_file, "r", encoding = "ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            if line == "":
                seq_set.append(" ".join(seq))
                seq_set_label.append(seq_label)
                seq_set_len.append(len(seq_label))
                seq = []
                seq_label = []
            else:
                if de:
                    tok, lemma, chunktag, postag, label = line.split()
                else:
                    tok, chunktag, postag, label = line.split()
                seq.append(tok)
                seq_label.append(labels_map[label])
    return [seq_set, seq_set_label, seq_set_len]

import unicodedata

def load_crosslingual_embeddings(input_file, vocab, max_vocab_size=20000, is_do_test=False):
    print('opening embedding ', 'data/' + input_file)
    embeddings = list(open('data/' + input_file, "r").readlines())
    pre_w2v = {}
    emb_size = 0
    for emb in embeddings:
#        emb = unicodedata.normalize("NFKD", emb)

        parts = emb.strip().split()
        sidx=1

#        if (emb[0] != ' ' and emb_size != (len(parts) - 1)) or (emb[0] == ' ' and emb_size != len(parts)):

        w = ' '
        if (emb_size != (len(parts) - 1)):
            if emb_size == 0:
                emb_size = len(parts) - 1
            else:
                if unicodedata.normalize("NFKD", emb)[0] == ' ':
                    sidx=0
                else:
                    print('ignoring line ', emb)
                    continue
        else:
            w = parts[0]


        vec = []
        try:
            for i in range(sidx, len(parts)):
                vec.append(float(parts[i]))
        except:
            print(emb)
            print(len(parts))
        # print w, vec
        pre_w2v[w] = vec

    n_dict = len(vocab._mapping)
    vocab_w2v = [None] * n_dict
    # vocab_w2v[0]=np.random.uniform(-0.25,0.25,100)
    for w, i in vocab._mapping.items():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        else:
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))

    cur_i = len(vocab_w2v)
    if len(vocab_w2v) > max_vocab_size:
        print("Vocabulary size is larger than", max_vocab_size)
        raise SystemExit
    while cur_i < max_vocab_size:
        cur_i += 1
        padding = [0] * emb_size
        vocab_w2v.append(padding)
    print("Vocabulary", n_dict, "Embedding size", emb_size)
    return vocab_w2v


def data2sents(X, Y):
    data = []
    for i in range(len(Y)):
        sent = []
        text = X[i]
        items = text.split()
        for j in range(len(Y[i])):
            sent.append((items[j], str(Y[i][j])))
        data.append(sent)
    return data
