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


random_state = 11122

#load the data set

X,Y = load_boston(return_X_y=True)

X = X.astype(np.float32)
Y = Y.astype(np.float32)
Y = Y.reshape((Y.shape[0], 1))

input_dimension = X.shape[1]

#define the architecture

nr_of_epochs = 700 # adam
name = 'cp'
learning_rate = 0.0001 # adam
dimensions =(input_dimension ,50 ,'dropout')
activations =('input' ,'rectify' ,('dropout' ,0.3))
batch_size = 5
error_type = 'squared_error'
constant_c = 0
constant_alpha = 0
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

seed_list = [41631779, 68230795, 49590161, 54924514, 71045363, 66211075, 61684656, 50278163, 20937475, 46627605, 46397998,
             31878001, 69581679, 23089777, 62161373, 25311839, 80691142, 78448556, 80439117, 85557277, 78373174, 55052651,
             88729554, 93387946, 90286980, 15157026, 32768491, 12545534, 21375697, 35006917, 72287407, 63216512, 79603427,
             94805126, 83292164, 28393302, 32477506, 14487582, 82539449, 70189663, 80556908, 62494106, 54818891, 21108579,
             55669982, 72039172, 17086712, 25450472, 89338559, 48416038]

rmse_list = []
auc_list_c = []
auc_list = []

cv_counter = 1
for random_state in seed_list:

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05,
                                                            random_state=random_state)

        # add outlier

        add_outlier = True
        percentage = 0.05

        if add_outlier:
            n = int(percentage * X_train.shape[0])
            np.random.seed(random_state)
            index = np.random.choice(X_train.shape[0], size=n, replace=False)
            np.random.seed()
            y = sp.stats.norm(loc=21, scale=111).rvs(n, random_state=134245)

            for i in range(n):
                print('changed', float(Y_train[index[i]]), 'to', y[i])
                Y_train[index[i]] = y[i]

        # normalize input and output

        StSc = StandardScaler()

        X_train = StSc.fit_transform(X_train)
        X_test = StSc.transform(X_test)

        StSc_Y = StandardScaler()

        Y_train = StSc_Y.fit_transform(Y_train.reshape((-1, 1)))

        #create the network

        cp_network = CPNetwork(architecture)

        #fit the network

        cp_network = cp_network.fit(X_train, Y_train)

        #predict the test set

        #predict uncorrected

        p, alpha, scale = cp_network.predict(X_test, use_correction=False)

        p = StSc_Y.inverse_transform(p)

        alpha[alpha<=2] = 2.0001 #get rid of infinite variances
        v = scale*alpha/(alpha-2)

        #predict prognostic

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

        auc_list.append(auc)
        auc_list_c.append(auc_c)
        rmse_list.append(rmse)


        #print results
        print("")
        print('Cross-validation', cv_counter)
        print('Student RMSE', rmse)
        print('GCP RMSE (same as Student)', rmse_c)
        print("")
        print('Student AUC', auc)
        print('GCP AUC', auc_c)
        print("")
        
        cv_counter = cv_counter + 1

print("")        
print('Averaged scores:')
print('Student and GCP RMSE', np.mean(np.asarray(rmse_list)))
print('Student AUC', np.mean(np.asarray(auc_list)))
print('GCP AUC', np.mean(np.asarray(auc_list_c)))
