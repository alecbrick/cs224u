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
            **kwargs):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.hidden_dim_v = int(np.sqrt(embed_dim)) ** 2
        self.max_length = max_length
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        super(TfTreeRNNClassifier, self).__init__(hidden_dim=int(np.sqrt(embed_dim)), **kwargs)
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
        self.input_lens = tf.placeholder(
            tf.int32, [None])
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        self.feats = tf.nn.embedding_lookup(
            self.embedding, self.inputs)

        # Need to do lift
        # H = tanh(W_lift*c + b_lift)

        # First, we define W_lift. This is actually a 3-D tensor, since it
        # lifts our input vectors into a sqrt(d)-by-sqrt(d) matrix.
        # Initialize with Xavier initialization, then shape into 3D.
        self.W_lift = tf.reshape(self.weight_init(
            self.embed_dim, int(self.hidden_dim ** 2), 'W_lift'), 
            [self.embed_dim, self.hidden_dim, self.hidden_dim])
        self.b_lift = tf.reshape(self.bias_init(
            int(self.hidden_dim ** 2), 'b_lift'),
            [self.hidden_dim, self.hidden_dim])

        self.lifted_feats = tf.nn.tanh(tf.tensordot(self.feats, self.W_lift, [[2], [0]]) / 100 + self.b_lift)
        
        # 224D
        self.is_leaf_t = tf.transpose(self.is_leaf)
        self.left_children_t = tf.transpose(self.left_children)
        self.right_children_t = tf.transpose(self.right_children)
        self.lifted_feats_t = tf.transpose(self.lifted_feats, [1, 0, 2, 3])

        # For node combination
        self.W_lstm = self.weight_init(
            2 * self.hidden_dim_v, 4 * self.hidden_dim_v, 'W_lstm')
        self.b_lstm = self.bias_init(
            4 * self.hidden_dim_v, 'b_lstm')
        self.W_comb = self.weight_init(self.hidden_dim, self.hidden_dim, 'W_comb')
        self.b_comb = self.weight_init(self.hidden_dim, self.hidden_dim, 'b_comb')
        #self.b_comb2 = self.weight_init(self.hidden_dim, self.hidden_dim, 'b_comb2')
        # maybe xavier init here
        x = np.sqrt(6.0/self.hidden_dim_v)
        #self.c_init = tf.Variable(tf.random_uniform(tf.shape(self.lifted_feats_t[0]), minval=-x, maxval=x), name="c_init")

        node_tensors = tf.TensorArray(tf.float32, size=1, 
                #element_shape=(2, self.inputs.shape[0], self.hidden_dim, self.hidden_dim),
                dynamic_size=True, clear_after_read=False, infer_shape=True)
        # So TF doesn't complain. We're not going to use this value.
        #node_tensors = node_tensors.write(0, [self.lifted_feats_t[0], self.lifted_feats_t[0]])
        #x = node_tensors.gather([0])
        


        # From 224D github
        # Loop through the tensors, combining them
        def loop_body(node_tensors, i):
            node_is_leaf = tf.gather(self.is_leaf_t, i)
            left_child = tf.gather(self.left_children_t, i)
            right_child = tf.gather(self.right_children_t, i)
            # batchy
            # keep track of [c, H]
            node_tensor = tf.where(
                node_is_leaf,
                tf.stack([tf.zeros_like(self.lifted_feats_t[0]), tf.gather(self.lifted_feats_t, i)], axis=1),
                # the things i do for batching
                tf.cond(tf.equal(i, 0),
                    lambda: tf.stack([tf.zeros_like(self.lifted_feats_t[0]), tf.gather(self.lifted_feats_t, i)], axis=1),
                    lambda: self.combine_children(node_tensors.gather(left_child),
                                     node_tensors.gather(right_child))))
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
        last_H = self.get_last_val(last_pair) # allow for inheritance
        self.last = tf.reshape(last_H, [-1, self.hidden_dim_v])
        self.W_hy = self.weight_init(
            self.hidden_dim_v, self.output_dim, 'W_hy')
        self.b_y = self.bias_init(self.output_dim, 'b_y')
        self.model = tf.matmul(self.last, self.W_hy) + self.b_y
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
        words, is_leaf, left_children, right_children, input_lens = self._convert_X(X)
        return {
            self.inputs: words,
            self.is_leaf: is_leaf,
            self.left_children: left_children,
            self.right_children: right_children,
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
        words, is_leaf, left_children, right_children, input_lens = self._convert_X(X)
        return {
            self.inputs: words,
            self.is_leaf: is_leaf,
            self.left_children: left_children,
            self.right_children: right_children,
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
        new_X = np.zeros((len(words), self.max_length), dtype='int')
        
        # For padding purposes.
        input_arrays = [words, is_leaf, left_children, right_children]
        padded_inputs = np.zeros((len(input_arrays), len(words), self.max_length), dtype='int')
        # ex_lengths = []
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index['unk']
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

        return padded_inputs[0], padded_inputs[1], padded_inputs[2], padded_inputs[3], input_lens

    def get_optimizer(self):
        return tf.train.AdamOptimizer(
            self.eta).minimize(self.cost)

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
        h_c_left = tf.reshape(left_H, [-1, self.hidden_dim_v])
        h_c_right = tf.reshape(right_H, [-1, self.hidden_dim_v])
        c_left = tf.reshape(left_C, [-1, self.hidden_dim_v])
        c_right = tf.reshape(right_C, [-1, self.hidden_dim_v])

        # LSTM calculations
        h_concat = tf.concat([h_c_left, h_c_right], 1)
        z_lstm = tf.matmul(h_concat, self.W_lstm) + self.b_lstm
        sig_lstm = tf.sigmoid(z_lstm)
        i, f_l, f_r, o = tf.split(sig_lstm, 4, axis=1)
        H_inner = tf.matmul(right_H, left_H)
        H_cand = tf.tanh(tf.tensordot(H_inner, self.W_comb, [2, 0]) + self.b_comb)
        # H_cand is now batch - size [?, d, d]
        # H_cand = tf.tanh(tf.matmul(right_H, H_inner) + self.b_comb2)
        g = tf.reshape(H_cand, [-1, self.hidden_dim_v])
        c = f_l * c_left + f_r * c_right + i * g
        H = tf.reshape(o * c, [-1, self.hidden_dim, self.hidden_dim])
        C = tf.reshape(c, [-1, self.hidden_dim, self.hidden_dim])
        return tf.stack([C, H], axis=1)

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
