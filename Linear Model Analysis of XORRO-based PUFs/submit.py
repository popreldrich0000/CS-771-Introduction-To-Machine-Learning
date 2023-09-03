import numpy as np
import sklearn
from sklearn import linear_model

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
  x_train = {}
  y_train = {}
  for i in range(16):
      for j in range(16):
          x_train[str(i)+str(j)]=[]
          y_train[str(i)+str(j)]=[]
  for row in Z_train:
    res_y = row[72]
    num1 = row[64]*8+row[65]*4+row[66]*2+row[67]
    num2 = row[68]*8+row[69]*4+row[70]*2+row[71]
    if num1 > num2:
      num1, num2 = num2, num1
      res_y = 1 - res_y
    num1 = int(num1)
    num2 = int(num2)
    x_train[str(num1)+str(num2)].append(row[:64])
    y_train[str(num1)+str(num2)].append(res_y)
  model = {}
  for i in range(16):
    for j in range(16):
      idx = str(i)+str(j)
      if len(x_train[idx])!=0:
        lrgs = linear_model.LogisticRegression()
        model[str(i)+str(j)]=lrgs.fit(x_train[idx],y_train[idx])

  return model
  



  # for i in range(16):
  #   for j in range(16):
  #     if(i<j):

  #       y = data_train[str(i)+" - "+str(j)]['72'].to_numpy()
  #       r = data_train[str(i)+" - "+str(j)].shape[0]
  #       X = data_train[str(i)+" - "+str(j)][:r]
  #       X = X.drop(X.columns[[64,65,66,67,68,69,70,71,72,73]],axis=1).to_numpy() 
  #       model_trained =  fit_ml_algo(linear_model.LogisticRegression(), X, y, 5)
  #       model.append(model_trained)
  # return model			# Return the trained model


# model =  my_fit(train)



################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################
  ypred = []
  for row in X_tst:
    num1 = row[64]*8+row[65]*4+row[66]*2+row[67]
    num2 = row[68]*8+row[69]*4+row[70]*2+row[71]
    flag = 0
    if num1 > num2:
      flag = 1 - flag
      num1, num2 = num2, num1
    num1 = int(num1)
    num2 = int(num2)
    idx = str(num1)+str(num2)
    result = model[idx].predict([row[:64]])[0]
    if flag:
        result = 1 - result
    ypred.append(result)
  return np.array(ypred).reshape(1,-1)

