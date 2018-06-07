# Basically everything from the Jupyter notebook, but in a single runnable file.

import numpy as np
import os
import sst
import tf_trnn
import tf_lifted_trnn
import time
import utils

vsmdata_home = 'vsmdata'
glove_home = os.path.join(vsmdata_home, 'glove.6B')
glove_lookup = utils.glove2dict(
    os.path.join(glove_home, 'glove.6B.50d.txt'))

def main():
    train = sst.build_dataset(
            sst.train_reader, lambda x: x, sst.ternary_class_func, vectorizer=None, vectorize=False)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    X_assess = None
    y_assess = None
    # Assessment dataset using the training vectorizer:
    assess = sst.build_dataset(
        sst.dev_reader,
        lambda x: x,
        sst.ternary_class_func,
        vectorizer=train['vectorizer'],
        vectorize=False)
    X_assess, y_assess = assess['X'], assess['y']

    tree_vocab = ["unk"] + sst.get_vocab(map(lambda x: x.leaves(), X_train), 
                                         n_words=3000)
    embedding = np.asarray(
        [glove_lookup.get(k, glove_lookup["unk"]) for k in tree_vocab])
    model = tf_trnn.TfTreeRNNClassifier(
        tree_vocab, 
        eta=0.0001,
        batch_size=32,
        embed_dim=50,
        hidden_dim=50,
        max_length=200,
        max_iter=25,
        embedding=embedding,
        train_embedding=True)
    model.fit(X_train, y_train, X_assess=X_assess, y_assess=y_assess)
    end = time.time()

if __name__ == "__main__":
    main()
