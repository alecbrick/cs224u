import nltk
import numpy as np
import tensorflow as tf
import tf_model_base
import warnings
import importlib
import tf_trnn

__author__ = 'Alec Brickner'

warnings.filterwarnings('ignore', category=UserWarning)

importlib.reload(tf_trnn)

class TfLiftedTreeRNNClassifier(tf_trnn.TfTreeRNNClassifier):
    def __init__(self, 
            vocab, 
            embedding=None,
            embed_dim=50,
            max_length=20,
            train_embedding=True,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            hidden_dim=50,
            **kwargs):
        self.hidden_dim_v = int(np.sqrt(embed_dim)) ** 2
        super(TfLiftedTreeRNNClassifier, self).__init__(vocab, embedding, embed_dim, max_length, train_embedding, cell_class, hidden_dim=int(np.sqrt(embed_dim)), **kwargs)

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
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.max_length, self.output_dim])

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

        node_tensors = tf.TensorArray(tf.float32, size=self.max_length, 
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
        last_H = tf.reshape(self.get_last_val(last_pair), [-1, self.hidden_dim_v]) # allow for inheritance

        hidden_vals = tf.reshape(node_tensors.stack()[:, :, 1], [self.max_length, -1, self.hidden_dim_v])

        self.W_hy = self.weight_init(
            self.hidden_dim_v, self.output_dim, 'W_hy')
        self.b_y = self.bias_init(self.output_dim, 'b_y')
        tiled_W_hy = tf.reshape(tf.tile(self.W_hy, [self.max_length, 1]), [self.max_length, self.hidden_dim_v, self.output_dim])

        self.model = tf.transpose(tf.matmul(hidden_vals, tiled_W_hy) + self.b_y, [1, 0, 2])
        self.last = tf.matmul(last_H, self.W_hy) + self.b_y 
        self.node_tensors = node_tensors

    def _onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.max_length, self.output_dim))
        for i, labels in enumerate(y):
            for j, cls in enumerate(labels):
                y_[i][j][classmap[cls]] = 1.0
        return y_

    def prepare_output_data(self, y):
        """Handle the list of lists that is y.
        First, get the set of labels that are _not_ None.
        """
        flattened_y = [elem for label in y for elem in label]
        self.classes = sorted(set(flattened_y))
        self.output_dim = len(self.classes)
        y = self._onehot_encode(y)
        return y


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
