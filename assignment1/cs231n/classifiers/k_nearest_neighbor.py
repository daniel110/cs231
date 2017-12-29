import numpy as np
from past.builtins import xrange

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #DF
        dists[i,j] = np.sqrt(np.sum((X[i]-self.X_train[j])**2))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #DF
      """
      def l2(test, trained):
        print(test.shape)
        print("Trained {0}".format(trained.shape))
        return np.linalg.norm(test-trained)

      l2_vect = np.vectorize(l2, excluded=['test'])
      dists[i, :] = l2_vect(test=X[i], trained=self.X_train[1:3])
      """

      squared = (X[i] - self.X_train) ** 2
      dists[i, :] = np.sqrt(np.sum(squared, 1))# sum by each array


      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #DF - from https://stackoverflow.com/questions/32856726/memory-efficient-l2-norm-using-python-broadcasting
    x2 = np.sum(X ** 2, axis=1).reshape((num_test, 1))
    y2 = np.sum(self.X_train ** 2, axis=1).reshape((1, num_train))
    xy = X.dot(self.X_train.T)  # shape is (m, n)
    dists = np.sqrt(x2 + y2 - 2 * xy)  # shape is (m, n)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #df
      sorted_indexs = np.argsort(dists[i])
      for j in xrange(k):
        trained_img_lable = self.y_train[sorted_indexs[j]]
        closest_y.append(trained_img_lable)

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #sort the lables
      closest_y.sort()
      if len(closest_y) == 0:
        y_pred[i] = "no closest images"
        continue
      
      current_num = 0
      current_lable = closest_y[0]
      max_num = current_num
      max_lable = current_lable
      # iterate over all the lables
      for lable in closest_y:
        if current_lable != lable:
          current_num = 1
          current_lable = lable
        else:
          current_num = current_num + 1
          if current_num > max_num:
            max_num = current_num
            max_lable = lable
      
      y_pred[i] = max_lable
         
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred


"""
# Last box code. Run different k's and look for the best using cross_validation I wrote.

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
def leaveFoldOut(folds, fold_index):
    inner_shape = folds[0].shape
    if len(inner_shape) > 1:
        allfolds = np.zeros(shape=((0,inner_shape[1])))
    else:
        allfolds = np.zeros(shape=(0))

    c_i = 0
    for i in range(len(folds)):
        if fold_index != i:
            allfolds = np.append(allfolds, folds[i], axis=0)

    return allfolds

def accurecy(y_test_pred, y_test):
    num_test = y_test.shape[0]
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test

    return accuracy

for k in k_choices:
    accs_k = []
    for i in range(num_folds):
        c_x_train = leaveFoldOut(X_train_folds, i)
        c_y_train = leaveFoldOut(y_train_folds, i)

        c_x_test = X_train_folds[i]
        c_y_test = y_train_folds[i]

        model = KNearestNeighbor()
        model.train(c_x_train, c_y_train)
        c_y_pred = model.predict(c_x_test,k)

        accs_k.append(accurecy(c_y_pred, c_y_test))

    k_to_accuracies[k] = accs_k

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    #DF added sum_acc
    sum_acc = 0
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
        sum_acc += accuracy

    print("!! Total Acc {0} !!\n".format(sum_acc/len(k_to_accuracies[k])))
"""