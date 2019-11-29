from linear_gae.layers import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

"""
Disclaimer: functions and classes defined from lines 15 to 126 in this file 
come from tkipf/gae original repository on Graph Autoencoders. Classes from
line 129 correspond to Linear and Deep models from our paper.
"""


class Model(object):
    """ Model base class"""
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    """
    Standard Graph Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder and inner product decoder
    """
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = FLAGS.hidden,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = FLAGS.hidden,
                                       output_dim = FLAGS.dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)


class GCNModelVAE(Model):
    """
    Standard Graph Variational Autoencoder from Kipf and Welling (2016),
    with 2-layer GCN encoder, Gaussian distributions and inner product decoder
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = FLAGS.hidden,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = FLAGS.hidden,
                                       output_dim = FLAGS.dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden)

        self.z_log_std = GraphConvolution(input_dim = FLAGS.hidden,
                                          output_dim = FLAGS.dimension,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging)(self.hidden)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.dimension]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z)


class LinearModelAE(Model):
    """
    Linear Graph Autoencoder, as defined in Section 3 of NeurIPS 2019 workshop paper,
    with linear encoder and inner product decoder
    """
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(LinearModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.z_mean = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = FLAGS.dimension,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)


class LinearModelVAE(Model):
    """
    Linear Graph Variational Autoencoder, as defined in Section 3 of
    NeurIPS 2019 workshop paper, with Gaussian distributions, linear
    encoders for mu and sigma, and inner product decoder
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(LinearModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):

        self.z_mean = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = FLAGS.dimension,
                                             adj = self.adj,
                                             features_nonzero=self.features_nonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_log_std = GraphConvolutionSparse(input_dim = self.input_dim,
                                                output_dim = FLAGS.dimension,
                                                adj = self.adj,
                                                features_nonzero = self.features_nonzero,
                                                act = lambda x: x,
                                                dropout = self.dropout,
                                                logging = self.logging)(self.inputs)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.dimension]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z)


class DeepGCNModelAE(Model):
    """
    "Deep" Graph Autoencoder from Section 4 of NeurIPS 2019 workshop paper,
    with 3-layer GCN encoder, and inner product decoder
    """
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(DeepGCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = FLAGS.hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim = FLAGS.hidden,
                                        output_dim = FLAGS.hidden,
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout,
                                        logging = self.logging)(self.hidden1)

        self.z_mean = GraphConvolution(input_dim = FLAGS.hidden,
                                       output_dim = FLAGS.dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden2)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)


class DeepGCNModelVAE(Model):
    """
    "Deep" Graph Variational Autoencoder, from Section 4 of NeurIPS 2019
    workshop paper, with Gaussian distributions, 3-layer GCN encoders for
    mu and sigma, and inner product decoder
    """
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(DeepGCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = FLAGS.hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim = FLAGS.hidden,
                                        output_dim = FLAGS.hidden,
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout,
                                        logging = self.logging)(self.hidden1)

        self.z_mean = GraphConvolution(input_dim = FLAGS.hidden,
                                       output_dim = FLAGS.dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden2)

        self.z_log_std = GraphConvolution(input_dim = FLAGS.hidden,
                                          output_dim = FLAGS.dimension,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging)(self.hidden2)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.dimension]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z)