import nltk
import numpy as np
import tensorflow as tf
import tf_model_base
import warnings
import importlib

__author__ = 'Alec Brickner'

warnings.filterwarnings('ignore', category=UserWarning)

importlib.reload(tf_model_base)

class TfTreeRNNClassifier(tf_model_base.TfModelBase):
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            max_length=20,
            train_embedding=True,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            hidden_dim=50,
            use_phrases=False,
            reg=0.001,
            **kwargs):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        self.use_phrases = use_phrases
        self.reg = reg
        super(TfTreeRNNClassifier, self).__init__(hidden_dim, **kwargs)
        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']

    def build_graph(self):
        self._define_embedding()
        # hacky but w/e
        try:
            self.embed_dim = self.embed_dim.value
        except:
            pass

        # The inputs
        self.inputs = tf.placeholder(
            tf.int32, [None, self.max_length])
        self.is_leaf = tf.placeholder(
            tf.bool, [None, self.max_length])
        self.left_children = tf.placeholder(
            tf.int32, [None, self.max_length])
        self.right_children = tf.placeholder(
            tf.int32, [None, self.max_length])
        self.is_node = tf.placeholder(
            tf.bool, [None, self.max_length])
        self.input_lens = tf.placeholder(
            tf.int32, [None])
        output_shape = [None, self.output_dim]
        if self.use_phrases:
            output_shape = [None, self.max_length, self.output_dim]
        self.outputs = tf.placeholder(
            tf.float32, shape=output_shape)

        self.feats = tf.nn.embedding_lookup(
            self.embedding, self.inputs)

        # 224D
        self.is_leaf_t = tf.transpose(self.is_leaf)
        self.left_children_t = tf.transpose(self.left_children)
        self.right_children_t = tf.transpose(self.right_children)
        self.feats_t = tf.transpose(self.feats, [1, 0, 2])

        # For node combination
        self.W_lstm = self.weight_init(
            2 * self.hidden_dim, 5 * self.hidden_dim, 'W_lstm')
        self.b_lstm = self.bias_init(
            5 * self.hidden_dim, 'b_lstm')

        # maybe xavier init here
        x = np.sqrt(6.0/self.hidden_dim)
        #self.c_init = tf.Variable(tf.random_uniform(tf.shape(self.lifted_feats_t[0]), minval=-x, maxval=x), name="c_init")

        node_tensors = tf.TensorArray(tf.float32, size=self.max_length, 
                #element_shape=(2, self.inputs.shape[0], self.hidden_dim, self.hidden_dim),
                dynamic_size=False, clear_after_read=False, infer_shape=True)
        # So TF doesn't complain. We're not going to use this value.
        #node_tensors = node_tensors.write(0, [self.lifted_feats_t[0], self.lifted_feats_t[0]])
        #x = node_tensors.gather([0])
        
        max_len = tf.reduce_max(self.input_lens)

        # From 224D github
        # Loop through the tensors, combining them
        def loop_body(node_tensors, i):
            node_is_leaf = tf.gather(self.is_leaf_t, i)
            left_child = tf.gather(self.left_children_t, i)
            right_child = tf.gather(self.right_children_t, i)
            leaf_tensor = tf.stack([tf.zeros_like(self.feats_t[0]), tf.gather(self.feats_t, i)], axis=1)
            # batchy
            # keep track of [c, H]
            node_tensor = tf.cond(
                tf.less(i, max_len),
                lambda: tf.where(
                    node_is_leaf,
                    leaf_tensor,
                    # the things i do for batching
                    tf.cond(tf.equal(i, 0),
                        lambda: leaf_tensor,
                        lambda: self.combine_children(
                                         node_tensors.gather(left_child),
                                         node_tensors.gather(right_child)))),
                lambda: leaf_tensor)
            node_tensors = node_tensors.write(i, node_tensor)
            i = tf.add(i, 1)
            return node_tensors, i

        # while less than #nodes
        loop_cond = lambda node_tensors, i: \
            tf.less(i, tf.reduce_max(self.input_lens))
        # loop thru 
        node_tensors, _ = tf.while_loop(loop_cond, loop_body, [node_tensors, 0],
                                            parallel_iterations=1)

        # Get the last [C, H], and retrieve H from that.
        last_pair = node_tensors.gather(self.input_lens - 1)
        last = self.get_last_val(last_pair) # allow for inheritance

        hidden_vals = node_tensors.stack()[:, :, 1]
        #hidden_vals = tf.reshape(tf.transpose(hidden_vals, [1, 0, 2]), [-1, self.max_length, self.hidden_dim])

        self.W_hy = self.weight_init(
            self.hidden_dim, self.output_dim, 'W_hy')
        self.b_y = self.bias_init(self.output_dim, 'b_y')
        tiled_W_hy = tf.reshape(tf.tile(self.W_hy, [self.max_length, 1]), [self.max_length, self.hidden_dim, self.output_dim])
        self.model = tf.transpose(tf.matmul(hidden_vals, tiled_W_hy) + self.b_y, [1, 0, 2])
        self.last = tf.matmul(last, self.W_hy) + self.b_y
        self.node_tensors = node_tensors

    def train_dict(self, X, y):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, , and gets the true length of each example
        and passes it to `fit` as well. `y` is fed to `outputs`.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        words, is_leaf, left_children, right_children, is_node, input_lens = self._convert_X(X)
        if self.use_phrases:
            for b in range(len(y)):
                for ex in range(len(y[b])):
                    if sum(y[b][ex]) == 0:
                        is_node[b][ex] = 0 # DON'T use these!
        return {
            self.inputs: words,
            self.is_leaf: is_leaf,
            self.left_children: left_children,
            self.right_children: right_children,
            self.is_node: is_node,
            self.input_lens: input_lens,
            self.outputs: y}

    def test_dict(self, X):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, and gets the true length of each example and
        passes it to `fit` as well.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        words, is_leaf, left_children, right_children, is_node, input_lens = self._convert_X(X)
        return {
            self.inputs: words,
            self.is_leaf: is_leaf,
            self.left_children: left_children,
            self.right_children: right_children,
            self.is_node: is_node,
            self.input_lens: input_lens
        }

    def _define_embedding(self):
        """Build the embedding matrix. If the user supplied a matrix, it
        is converted into a Tensor, else a random Tensor is built. This
        method sets `self.embedding` for use and returns None.
        """
        if type(self.embedding) == type(None):
            self.embedding = tf.Variable(
                tf.random_uniform(
                    [self.vocab_size, self.embed_dim], -1.0, 1.0),
                trainable=self.train_embedding)
        else:
            self.embedding = tf.Variable(
                initial_value=self.embedding,
                dtype=tf.float32,
                trainable=self.train_embedding)
            self.embed_dim = self.embedding.shape[1]

    def _phrase_onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.max_length, self.output_dim))
        for i, labels in enumerate(y):
            for j, cls in enumerate(labels):
                if cls in classmap:
                    y_[i][j][classmap[cls]] = 1.0
        return y_

    def _onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = 1.0
        return y_

    def prepare_phrase_output_data(self, y):
        """Handle the list of lists that is y.
        First, get the set of labels that are _not_ None.
        """
        flattened_y = [labels[-1] for labels in y]
        self.classes = sorted(set(flattened_y))
        self.output_dim = len(self.classes)
        y = self._phrase_onehot_encode(y)
        return y

    def prepare_sentence_output_data(self, y):
        self.classes = sorted(set(y))
        self.output_dim = len(self.classes)
        y = self._onehot_encode(y)
        return y
    
    def prepare_output_data(self, y):
        if self.use_phrases:
            return self.prepare_phrase_output_data(y)
        else:
            return self.prepare_sentence_output_data(y)

    def _convert_X(self, X):
        """ ((my dog) (is cool)
        is_leaf: [True, True, True, True, False, False, False]
        [0, 1, 2, 3, -1, -1, -1]
        left_children: [-1, -1, -1, -1, 0, 2, 4]
        right_children: [-1, -1, -1, -1, 1, 3, 5]
        We might be able to accomplish this with a postorder traversal.
        
        X: NLTK tree.
        out: is_leaf, words, 
        """

        is_leaf, words, left_children, right_children, input_lens = \
            zip(*map(lambda x: traverse(x, 0), X))
        is_node = list(map(lambda tree: list(map(lambda x: not x, tree)), is_leaf))
        new_X = np.zeros((len(words), self.max_length), dtype='int')
        
        # For padding purposes.
        input_arrays = [words, is_leaf, left_children, right_children, is_node]
        padded_inputs = np.zeros((len(input_arrays), len(words), self.max_length), dtype='int')
        # ex_lengths = []
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index.get('unk', index['$UNK'])
        # oh you gotta pad your inputs =/
        # And cut them off if they're too long!
        # But that's super difficult for trees, since they're so
        # structure-dependent. Oh well. I'll come to that later. 
        for i in range(new_X.shape[0]):
            # ex_lengths.append(len(words[i]))
            for j in range(len(input_arrays)):
                vals = input_arrays[j][i][-self.max_length:]
                if j == 0:
                    vals = [index.get(w, unk_index) for w in vals]
                temp = np.zeros((self.max_length,), dtype='int')
                temp[0: len(vals)] = vals
                padded_inputs[j][i] = temp

        return padded_inputs[0], padded_inputs[1], padded_inputs[2], padded_inputs[3], padded_inputs[4], input_lens

    def get_regularization(self):
        return self.reg * (
                tf.nn.l2_loss(self.W_hy) + tf.nn.l2_loss(self.W_lstm))

    def get_cost_function(self, **kwargs):
        if self.use_phrases:
            # Calculation is based on all phrase nodes.
            labels = tf.boolean_mask(self.outputs, self.is_node)
            logits = tf.boolean_mask(self.model, self.is_node)
        else:
            # Calculation is based on the sentence node.
            labels = self.outputs
            logits = self.last

        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=labels)
        return tf.reduce_mean(softmax) + self.get_regularization()


    def get_optimizer(self):
        return tf.train.AdamOptimizer(
            self.eta).minimize(self.cost)

    def predict_proba(self, X):
        return self.sess.run(
            self.last, feed_dict=self.test_dict(X))

    # Because there's no generalized diag operation.
    # Why is there no generalized diag operation? Sad state of affairs!
    def outer_diag(self, tensor):
        trans = tf.transpose(tensor)
        diag = tf.matrix_diag_part(trans)
        return tf.transpose(diag)

    # From 224D github
    # probably bad. update later
    # [batch] + left_child_dims
    # oh heck we need to keep track of state!
    def combine_children(self, left_tensor, right_tensor):
        left_C = self.outer_diag(left_tensor)[:, 0]
        left_H = self.outer_diag(left_tensor)[:, 1]
        right_C = self.outer_diag(right_tensor)[:, 0]
        right_H = self.outer_diag(right_tensor)[:, 1]

        # LSTM calculations
        h_concat = tf.concat([left_H, right_H], 1)
        z_lstm = tf.matmul(h_concat, self.W_lstm) + self.b_lstm
        z_i, z_fl, z_fr, z_o, z_g = tf.split(z_lstm, 5, axis=1)
        i = tf.sigmoid(z_i)
        f_l = tf.sigmoid(z_fl)
        f_r = tf.sigmoid(z_fr)
        o = tf.sigmoid(z_o)
        g = tf.tanh(z_g)
        # H_cand is now batch - size [?, d, d]
        # H_cand = tf.tanh(tf.matmul(right_H, H_inner) + self.b_comb2)
        c = f_l * left_C + f_r * right_C + i * g
        h = o * c 
        return tf.stack([c, h], axis=1)

    def get_last_val(self, last_val):
        last_H = self.outer_diag(last_val)[:, 1]
        return last_H


def traverse(tree, i):
    is_leaf = []
    words = []
    left_children = []
    right_children = []
    childs = []
    if type(tree) == nltk.tree.Tree:
        if len(tree) > 1:
            for subtree in tree:
                il, wds, lc, rc, i = traverse(subtree, i)
                is_leaf += il
                words += wds
                left_children += lc
                right_children += rc
                childs.append(i - 1)
            leaf = False
            left = childs[0]
            right = childs[1]
            word = 0 # We'll do 0 instead of -1 because TF doesn't like -1.
                     # We'll need to ensure that this doesn't update <unk> w.
        else:
            leaf = True
            left = 0 # Same here.
            right = 0 # And here. 
            word = tree[0]
    else:
        leaf = True
        left = 0 # And here too!
        right = 0 # Also here.
        word = tree
    is_leaf.append(leaf)
    words.append(word)
    left_children.append(left)
    right_children.append(right)
    i += 1
    return is_leaf, words, left_children, right_children, i
