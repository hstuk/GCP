import numpy as np
import scipy as sp
from frozendict import frozendict
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from CPNetwork import CPArchitecture, CPNetwork

def area_under_curve(Y):

    X = np.arange(Y.shape[0])/Y.shape[0]
    X_shifted = X[1:]
    dx = X_shifted-X[0:-1]
    auc = 0.5*(np.sum(Y[0:-1]*dx)+np.sum(Y[1:]*dx))

    return auc

def error_remove_curve(Y):

    err_curve = []

    for i in range(Y.shape[0]):
        err_curve.append(np.sqrt(np.mean(Y[i:])))

    return np.asarray(err_curve)


random_state = 19759677

#load the data set
X,Y = load_boston(return_X_y=True)

X = X.astype(np.float32)
Y = Y.astype(np.float32)
Y = Y.reshape((Y.shape[0], 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05,
                                                      random_state=random_state)

input_dimension = X_train.shape[1]

#add outlier

add_outlier = True
percentage = 0.05

if add_outlier:
    n = int(percentage * X_train.shape[0])
    np.random.seed(random_state)
    index = np.random.choice(X_train.shape[0], size=n, replace=False)
    np.random.seed()
    y = sp.stats.norm(loc = 21, scale = 111).rvs(n, random_state = 1)

    for i in range(n):
        print('changed', float(Y_train[index[i]]),'to', y[i])
        Y_train[index[i]] = y[i]

#normalize input and output

StSc = StandardScaler()

X_train = StSc.fit_transform(X_train)
X_test = StSc.transform(X_test)

StSc_Y = StandardScaler()

Y_train = StSc_Y.fit_transform(Y_train.reshape((-1,1)))
# = StSc_Y.transform(Y_test)

#define the architecture
nr_of_epochs = 700  
name = 'cp'
learning_rate = 0.0001 
dimensions =(input_dimension, 50, 'dropout')
activations =('input', 'rectify', ('dropout' ,0.3))
batch_size = 5
error_type = 'squared_error'
constant_c = 0 # don't fix the (beta,nu) trajectory
constant_alpha = 0 # don't fix alpha
data_set = 'boston'
use_predictive = True
momentum = 0.9
optimizer_setup =frozendict({'name': 'adam', 'beta1' :0.9, 'beta2' :0.999})
# optimizer_setup=frozendict({'name': 'rmsprop', 'rho':0.5})
# optimizer_setup=frozendict({'name': 'nesterov_momentum', 'momentum':0.9})
weight_decay_alpha = 0.001
weight_decay_beta = 0.001
weight_decay_nu = 0.001
weight_decay_mu = 0.001

architecture = CPArchitecture(name=name, learning_rate=learning_rate, dimensions_mu = dimensions, dimensions_alpha=dimensions,
                              dimensions_beta = dimensions, dimensions_nu=dimensions, activations_mu=activations, activations_alpha = activations,
                              activations_beta = activations, activations_nu = activations, momentum=momentum, nr_of_epochs=nr_of_epochs,
                              batch_size=batch_size, error_type=error_type, constant_c=constant_c, constant_alpha= constant_alpha,
                              data_set=data_set, use_predictive=use_predictive, optimizer_setup=optimizer_setup, weight_decay_alpha=weight_decay_alpha,
                              weight_decay_beta=weight_decay_beta ,weight_decay_nu=weight_decay_nu, weight_decay_mu=weight_decay_mu,
                              use_one_network = False)

#create the network
cp_network = CPNetwork(architecture)

#fit the network
cp_network = cp_network.fit(X_train, Y_train)

#predict the test set

#predict Student (uncorrected)
p, deg_freedom, scale2 = cp_network.predict(X_test, use_correction=False)

p = StSc_Y.inverse_transform(p)

deg_freedom[deg_freedom<=2] = 2.0001 #get rid of infinite variances
v = scale2*deg_freedom/(deg_freedom-2)

#predict GCP (prognostic)
p_c, v_c = cp_network.predict(X_test, use_correction=True)
p_c = StSc_Y.inverse_transform(p_c)

#calculate rmse and auc
err = (p-Y_test)**2
err_c = (p_c-Y_test)**2

rmse = np.sqrt(np.mean(err))
rmse_c = np.sqrt(np.mean(err_c)) #overall rmse is always the same

#sort errors by uncertainty
sorted = np.argsort(v,axis=0)
sorted_c = np.argsort(v_c, axis=0)

err = err[sorted]
err = err[::-1]
err_c = err_c[sorted_c]
err_c = err_c[::-1]

#calculate error removed curve
erc = error_remove_curve(err)
erc_c = error_remove_curve(err_c)

#calculate auc
auc = area_under_curve(erc)
auc_c = area_under_curve(erc_c)

#print results
print("")
print('Student RMSE', rmse)
print('GCP RMSE (same as Student)', rmse_c)
print("")
print('Student AUC', auc)
print('GCP AUC', auc_c)
