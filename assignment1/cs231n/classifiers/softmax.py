import numpy as np
from random import shuffle
from past.builtins import xrange

# https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # scores -= scores.max()
    #exp
    scores_exp = np.exp(scores)
    # normalization
    scores_exp_normal = scores_exp / np.sum(scores_exp )
    #natural log and accumulate loss
    loss += (-1)*np.log(scores_exp_normal[y[i]])


    for j in range(num_classes):
      #for correct class: -(1 - possibility(j)) * dimention(i)
      #for incorrect class : possibility(j) * dimention(i)
      dW[:,j] += (scores_exp_normal[j] - (j == y[i]) ) * X[i]

    # dW[:, :] +=   X[i].reshape(-1,1).dot(scores_exp_normal.reshape(1,-1))
    # dW[:, y[i]] +=  - X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss +=  reg * np.sum(W * W)
  dW +=  2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  # scores -= scores.max()
  #exp
  scores_exp = np.exp(scores)
  # normalization
  scores_exp_normal = scores_exp / np.sum(scores_exp, axis=1).reshape(-1,1)
  #natural log and accumulate loss
  loss = np.sum(\
    (-1)*np.log(\
      scores_exp_normal[np.arange(0,num_train,1), y]\
      )\
    )

  # for i in xrange(num_train):
  #   dW[:, :] +=   X[i].reshape(-1,1).dot(scores_exp_normal.reshape(1,-1))
  #   dW[:, y[i]] +=  - X[i]

  dW[:, :] +=   X.T.dot(scores_exp_normal)
  binary_y = np.zeros(scores.shape)
  binary_y[np.arange(0,binary_y.shape[0],1),y] = 1
  dW[:, :] += (-1) * X.T.dot(binary_y)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss +=  reg * np.sum(W * W)
  dW +=  2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

