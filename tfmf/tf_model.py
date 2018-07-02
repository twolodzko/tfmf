
import numpy as np
import tensorflow as tf


class TFModel(object):
   # Define and initialize the TensorFlow model, its weights, initialize session and saver

   def __init__(self, shape, learning_rate, alpha, regularization_rate,
                implicit, loss, log_weights, fit_intercepts, optimizer,
                random_state=None):

      self.shape = shape
      self.learning_rate = learning_rate
      self.alpha = alpha
      self.regularization_rate = regularization_rate
      self.implicit = implicit
      self.loss = loss
      self.log_weights = log_weights
      self.fit_intercepts = fit_intercepts
      self.optimizer = optimizer
      self.random_state = random_state

      tf.set_random_seed(self.random_state)
      self._init_model_and_session()


   def _init_model_and_session(self):

      # the R (n, k) matrix is factorized to P (n, d) and Q (k, d) matrices
      n, k, d = self.shape
      self.graph = tf.Graph()

      with self.graph.as_default():

         with tf.name_scope('constants'):
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            regularization_rate = tf.constant(self.regularization_rate, dtype=tf.float32,
                                              name='regularization_rate')

         with tf.name_scope('inputs'):
            self.row_ids = tf.placeholder(tf.int32, shape=[None], name='row_ids')
            self.col_ids = tf.placeholder(tf.int32, shape=[None], name='col_ids')
            self.values = tf.placeholder(tf.float32, shape=[None], name='values')

            if self.implicit:
               # D[i,j] = 1 if R[i,j] > 0 else 0
               targets = tf.clip_by_value(self.values, 0, 1, name='targets')
               
               if self.log_weights:
                  data_weights = tf.add(1.0, alpha * tf.log1p(self.values), name='data_weights')
               else:
                  data_weights = tf.add(1.0, alpha * self.values, name='data_weights')
            else:
               targets = tf.identity(self.values, name='targets')
               data_weights = tf.constant(1.0, name='data_weights')

         with tf.name_scope('parameters'):
            
            if self.fit_intercepts:
               # b0
               self.global_bias = tf.get_variable('global_bias', shape=[], dtype=tf.float32,
                                                  initializer=tf.zeros_initializer())
               # bi
               self.row_biases = tf.get_variable('row_biases', shape=[n], dtype=tf.float32,
                                                 initializer=tf.zeros_initializer())
               # bj
               self.col_biases = tf.get_variable('col_biases', shape=[k], dtype=tf.float32,
                                                 initializer=tf.zeros_initializer())

            # P (n, d) matrix
            self.row_weights = tf.get_variable('row_weights', shape=[n, d], dtype=tf.float32,
                                               initializer = tf.random_normal_initializer(mean=0, stddev=0.01))

            # Q (k, d) matrix
            self.col_weights = tf.get_variable('col_weights', shape=[k, d], dtype=tf.float32,
                                               initializer = tf.random_normal_initializer(mean=0, stddev=0.01))

         with tf.name_scope('prediction'):
            
            if self.fit_intercepts:
               batch_row_biases = tf.nn.embedding_lookup(self.row_biases, self.row_ids, name='row_bias')
               batch_col_biases = tf.nn.embedding_lookup(self.col_biases, self.col_ids, name='col_bias')

            batch_row_weights = tf.nn.embedding_lookup(self.row_weights, self.row_ids, name='row_weights')
            batch_col_weights = tf.nn.embedding_lookup(self.col_weights, self.col_ids, name='col_weights')

            # P[i,:] * Q[j,:]
            weights = tf.reduce_sum(tf.multiply(batch_row_weights, batch_col_weights), axis=1, name='weights')

            if self.fit_intercepts:
               biases = tf.add(batch_row_biases, batch_col_biases)
               biases = tf.add(self.global_bias, biases, name='biases')
               linear_predictor = tf.add(biases, weights, name='linear_predictor')
            else:
               linear_predictor = tf.identity(weights, name='linear_predictor')

            if self.loss == 'logistic':
               self.pred = tf.sigmoid(linear_predictor, name='predictions')
            else:
               self.pred = tf.identity(linear_predictor, name='predictions')

         with tf.name_scope('loss'):
            
            l2_weights = tf.add(tf.nn.l2_loss(self.row_weights),
                                tf.nn.l2_loss(self.col_weights), name='l2_weights')
            
            if self.fit_intercepts:
               l2_biases = tf.add(tf.nn.l2_loss(batch_row_biases),
                                  tf.nn.l2_loss(batch_col_biases), name='l2_biases')
               l2_term = tf.add(l2_weights, l2_biases)
            else:
               l2_term = l2_weights
            
            l2_term = tf.multiply(regularization_rate, l2_term, name='regularization')

            if self.loss == 'logistic':
               loss_raw = tf.losses.log_loss(predictions=self.pred, labels=targets,
                                             weights=data_weights)
            else:
               loss_raw = tf.losses.mean_squared_error(predictions=self.pred, labels=targets,
                                                       weights=data_weights)         

            self.cost = tf.add(loss_raw, l2_term, name='loss')

         if self.optimizer == 'Ftrl':
            self.train_step = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.cost)
         else:
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

         self.saver = tf.train.Saver()

         init = tf.global_variables_initializer()

      # initialize TF session
      self.sess = tf.Session(graph=self.graph)
      self.sess.run(init)
      
      
   def train(self, rows, cols, values):          
      batch = {
         self.row_ids : rows,
         self.col_ids : cols,
         self.values : values
      }
      _, loss_value = self.sess.run(fetches=[self.train_step, self.cost],
                                    feed_dict=batch)
      return loss_value
   
   
   def predict(self, rows, cols):
      batch = {
         self.row_ids : rows,
         self.col_ids : cols
      }
      return self.pred.eval(feed_dict=batch, session=self.sess)
   
   
   def coef(self):
      if self.fit_intercepts:
         return self.sess.run(fetches={
            'global_bias' : self.global_bias,
            'row_bias' : self.row_biases,
            'col_bias' : self.col_biases,
            'row_weights' : self.row_weights,
            'col_weights' : self.col_weights
         })
      else:
         return self.sess.run(fetches={
            'row_weights' : self.row_weights,
            'col_weights' : self.col_weights
         })
   
   
   def save(self, path):
      self.saver.save(self.sess, path)
   
   
   def restore(self, path):
      self.saver.restore(self.sess, path)