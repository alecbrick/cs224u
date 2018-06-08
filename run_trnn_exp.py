# Basically everything from the Jupyter notebook, but in a single runnable file.

import numpy as np
import os
import sst
import tf_trnn
import tf_lifted_trnn
import time
import utils

vsmdata_home = 'glove.6B'
glove_home = vsmdata_home
glove_lookup_100 = utils.glove2dict(
    os.path.join(glove_home, 'glove.6B.100d.txt'))
glove_lookup_200 = utils.glove2dict(
    os.path.join(glove_home, 'glove.6B.200d.txt'))

def run_experiment(eta, embed, model, phrase):
    print ("===================================")
    print ("eta: %s, embed_dim: %s, model: %s" % (eta, embed, model))
    print ("===================================")

    if embed == 100:
        glove_lookup = glove_lookup_100
    else:
        glove_lookup = glove_lookup_200

    if model == "lifted":
        base_model = tf_lifted_trnn.TfLiftedTreeRNNClassifier
    else:
        base_model = tf_trnn.TfTreeRNNClassifier

    start = time.time()
    train = sst.build_dataset(
            sst.train_reader, lambda x: x, sst.binary_class_func, vectorizer=None, vectorize=False, subtree_labels=phrase)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    X_assess = None
    y_assess = None
    # Assessment dataset using the training vectorizer:
    assess = sst.build_dataset(
        sst.dev_reader,
        lambda x: x,
        sst.binary_class_func,
        vectorizer=train['vectorizer'],
        vectorize=False, subtree_labels=phrase)
    X_assess, y_assess = assess['X'], assess['y']

    tree_vocab = ["unk"] + sst.get_vocab(map(lambda x: x.leaves(), X_train), 
                                         n_words=5000)
    embedding = np.asarray(
        [glove_lookup.get(k, glove_lookup["unk"]) for k in tree_vocab])
    model = base_model(
        tree_vocab, 
        eta=eta,
        batch_size=16,
        embed_dim=embed,
        hidden_dim=embed,
        max_length=120,
        max_iter=10,
        embedding=embedding,
        train_embedding=True,
        use_phrases=phrase)
    model.fit(X_train, y_train, X_assess=X_assess, y_assess=y_assess)
    end = time.time()
    print("Time: %s" % (end - start))

def main():
    params = {
        "eta": [0.0001, 0.001, 0.01],
        "embed": [100],
        "model": ["lifted", "normal"]
    }
    for em in params["embed"]:
        for m in params["model"]:
            for e in params["eta"]:
                run_experiment(e, em, m, False)

if __name__ == "__main__":
    main()
