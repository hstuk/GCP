from collections import namedtuple

import lasagne
import numpy as np
import scipy
import theano
import theano.tensor as T
from scipy.optimize import root
from scipy.stats import t

import distributions
from optimizer_setup import create_updates

CPArchitecture = namedtuple('CPArchitecture', ['name','learning_rate', 'dimensions_mu', 'dimensions_alpha', 'dimensions_beta', 'dimensions_nu',
                                              'activations_mu', 'activations_alpha', 'activations_beta', 'activations_nu', 'momentum',
                                              'nr_of_epochs','batch_size','error_type','constant_c','constant_alpha','data_set','use_predictive',
                                               'optimizer_setup','use_one_network', 'weight_decay_mu','weight_decay_alpha','weight_decay_beta',
                                                'weight_decay_nu'])

CPArchitecture.__new__.__defaults__ = (None,) * len(CPArchitecture._fields)


precalc_alpha = True
res = 0.005
max_alpha = 40

def F(alpha,A):
     return (np.pi)**(1/2)*np.exp(alpha-A)*scipy.special.erfc((alpha-A)**(1/2)) / (2*(alpha-A)**(1/2))

def IdentityAalpha(alpha,A):
     return (2*alpha+1)*(1-2*(alpha-A)*F(alpha,A))-1

def funcA(alpha):
    return scipy.optimize.bisect(lambda x: IdentityAalpha(alpha,x),0,np.min([alpha,1]))

def funcSetA2(set_alpha):
     set_A=[]
     if hasattr(set_alpha, "__len__"):
         for al in set_alpha:
             set_A.append(funcA(al))
         return np.array(set_A)
     else:
         return funcA(set_alpha)

def funcSetA(set_alpha):
    if precalc_alpha:
        set_alpha[set_alpha > max_alpha] = max_alpha
        return a_table[np.asarray(set_alpha/res,np.int32)]
    else:
        return funcSetA2(set_alpha)

a_table = 0

if(precalc_alpha):
    a_table = np.arange(0,max_alpha+res,res)
    a_table = funcSetA2(a_table)

def calculateVariance(alpha, beta, nu):
    A = funcSetA(alpha)
    A = A.reshape(alpha.shape)
    return beta*(nu+1)/((alpha - A )*nu)


class CPNetwork:

    def create_layers(self, data_var, dimensions, activations, weights=None):
        last_layer = lasagne.layers.InputLayer(shape=(None,1,dimensions[0],1), input_var = data_var)
        param_index = 0
        for i in range(1,len(dimensions)):
            if activations[i][0] == 'dropout':
                last_layer = lasagne.layers.DropoutLayer(last_layer, p=activations[i][1])
            else:
                if activations[i] == 'tanh':
                    nonlin = lasagne.nonlinearities.tanh
                elif activations[i] == 'linear':
                    nonlin = lasagne.nonlinearities.linear
                elif activations[i] == 'rectify':
                    nonlin = lasagne.nonlinearities.rectify
                elif activations[i] == 'sigmoid':
                    nonlin = lasagne.nonlinearities.sigmoid
                elif activations[i] == 'softplus':
                    nonlin = T.nnet.softplus

                if(weights == None):
                    last_layer = lasagne.layers.DenseLayer(last_layer,num_units = dimensions[i],
                                nonlinearity=nonlin,W=lasagne.init.GlorotUniform())
                else:
                    last_layer = lasagne.layers.DenseLayer(last_layer,num_units = dimensions[i],
                                nonlinearity=nonlin,W=weights[2*param_index], b = weights[2*param_index+1])
                    param_index = param_index + 1

        return last_layer

    def __init__(self,architecture):


        self.architecture = architecture

        learning_rate = T.scalar('learning_rate')
        data_var = T.tensor4('inputs')
        target_var = T.matrix('targets')

        self.input_dimension = architecture.dimensions_mu[0]
        self.output_dimension = 1

        self.mu_output_layer = self.create_layers(data_var,architecture.dimensions_mu, architecture.activations_mu)

        weights = None

        if architecture.use_one_network == True:
            weights = lasagne.layers.get_all_params(self.mu_output_layer, trainable=True)


        self.alpha_output_layer = self.create_layers(data_var,architecture.dimensions_alpha, architecture.activations_alpha, weights)
        self.beta_output_layer = self.create_layers(data_var,architecture.dimensions_beta, architecture.activations_beta, weights)
        self.nu_output_layer = self.create_layers(data_var,architecture.dimensions_nu, architecture.activations_nu, weights)

        self.batch_size = architecture.batch_size
        self.number_of_epochs = architecture.nr_of_epochs
        self.use_predictive = architecture.use_predictive
        self.constant_alpha = architecture.constant_alpha

        network_mu = lasagne.layers.DenseLayer(self.mu_output_layer, num_units=1, nonlinearity=lasagne.nonlinearities.linear)

        max_nu = 0

        if(architecture.constant_c != 0):
            max_nu = root(lambda x: architecture.constant_c - x**2- 2/3*x**3, x0=architecture.constant_c).x*0.9999

        if(architecture.constant_c == 0):
            network_nu = lasagne.layers.DenseLayer(self.nu_output_layer, num_units=1, nonlinearity=theano.tensor.nnet.softplus)
        else:
            network_nu = lasagne.layers.DenseLayer(self.nu_output_layer, num_units=1, nonlinearity=lambda x: max_nu*lasagne.nonlinearities.sigmoid(x))

        mu_out=  lasagne.layers.get_output(network_mu, deterministic=False)
        nu_out =  lasagne.layers.get_output(network_nu, deterministic=False)

        mu_outd = lasagne.layers.get_output(network_mu, deterministic=True)
        nu_outd = lasagne.layers.get_output(network_nu, deterministic=True)


        if(architecture.constant_c == 0):
            network_beta = lasagne.layers.DenseLayer(self.beta_output_layer, num_units=1, nonlinearity=theano.tensor.nnet.softplus)
            beta_out =  lasagne.layers.get_output(network_beta, deterministic=False)
            beta_outd = lasagne.layers.get_output(network_beta, deterministic=True)



        if(architecture.constant_alpha == 0):
                network_alpha = lasagne.layers.DenseLayer(self.alpha_output_layer, num_units=1, nonlinearity= lambda x: theano.tensor.nnet.softplus(x)+0.1)
                alpha_out =  lasagne.layers.get_output(network_alpha, deterministic=False)
                alpha_outd =  lasagne.layers.get_output(network_alpha, deterministic=True)
        else:
                alpha_out = T.constant(architecture.constant_alpha)
                alpha_outd = T.constant(architecture.constant_alpha)

        if(architecture.constant_c != 0):
                nu_out2 = nu_out

                beta_out = T.sqrt(architecture.constant_c-nu_out2*nu_out2*(1+2/3*nu_out2))
                beta_outd = T.sqrt(architecture.constant_c-nu_outd*nu_outd*(1+2/3*nu_outd))
                beta_out = theano.gradient.disconnected_grad(beta_out)


        n_alpha, n_beta, n_nu, n_mu = distributions.posterior_nd_ng(alpha_out, beta_out, nu_out, mu_out, target_var)

        if(architecture.constant_alpha == 0):
                n_alpha3 = theano.gradient.disconnected_grad(n_alpha)
        else:
                n_alpha3 = T.constant(architecture.constant_alpha+1/2)

        n_beta3 = theano.gradient.disconnected_grad(n_beta)
        n_nu3 = theano.gradient.disconnected_grad(n_nu)
        n_mu3 = theano.gradient.disconnected_grad(n_mu)

        loss = distributions.kl_div_ng_ng(n_alpha3, n_beta3, n_nu3, n_mu3,alpha_out, beta_out, nu_out, mu_out)

        loss = lasagne.objectives.aggregate(loss)

        loss = loss + architecture.weight_decay_mu*lasagne.regularization.regularize_network_params(network_mu, lasagne.regularization.l2 )

        if(architecture.constant_alpha == 0):
                loss = loss + architecture.weight_decay_alpha*lasagne.regularization.regularize_network_params(network_alpha, lasagne.regularization.l2 )

        if(architecture.constant_c == 0):
                loss = loss + architecture.weight_decay_beta*lasagne.regularization.regularize_network_params(network_beta, lasagne.regularization.l2 )

        loss = loss + architecture.weight_decay_nu*lasagne.regularization.regularize_network_params(network_nu, lasagne.regularization.l2 )

        params_c = lasagne.layers.get_all_params(network_mu,trainable=True)
        params_c.extend(lasagne.layers.get_all_params(network_nu, trainable=True))


        if(architecture.constant_alpha == 0):
                params_c.extend(lasagne.layers.get_all_params(network_alpha, trainable=True))

        if(architecture.constant_c == 0):
                params_c.extend(lasagne.layers.get_all_params(network_beta, trainable=True))

        self.params_c = params_c

        self.learning_rate = architecture.learning_rate
        optimizer_config = dict(architecture.optimizer_setup)
        optimizer_config['learning_rate'] = learning_rate
        updates_both = create_updates(optimizer_config,loss, params_c)

        self.evaluate_posterior = theano.function([data_var,target_var], [n_alpha, n_beta, n_nu, n_mu])
        self.train_fn = theano.function([data_var,target_var,learning_rate], loss, updates=updates_both)

        self.predict_fn = theano.function([data_var], [alpha_outd, beta_outd, nu_outd, mu_outd])

        mu_pred, alpha_pred, sigma_pred = distributions.predictive_nd_ng(alpha_outd, beta_outd, nu_outd, mu_outd)
        self.predict_parameters = theano.function([data_var], [alpha_outd, beta_outd, nu_outd, mu_outd])
        self.predict_predictive_fn = theano.function([data_var], [mu_pred, alpha_pred, sigma_pred])

        self.loss_fn = theano.function([data_var, target_var], loss)


    def fit(self,X,y,learning_rate=None, nr_of_epochs=None):

        if nr_of_epochs == None:
            epochs = self.number_of_epochs
        else:
            epochs = nr_of_epochs

        if learning_rate == None:
            learning_rate = self.architecture.learning_rate

        number_of_batches, rest = divmod(X.shape[0], self.batch_size)
        for j in range(epochs):
            training_stream = np.random.choice(X.shape[0], (number_of_batches,self.batch_size), replace=False)

            loss = 0

            for sample_choice in training_stream:
                sample = X[sample_choice,:]
                y_sample = y[sample_choice]

                sample = np.asarray(sample,dtype=np.float32)
                inputs = sample.reshape(self.batch_size,1,self.input_dimension,1)

                target = np.asarray(y_sample, dtype=np.float32)
                targets = target.reshape(sample_choice.shape[0],self.output_dimension)

                loss = loss+self.train_fn(inputs,targets,learning_rate)
            if j%50 == 0:
                print('nr of epochs',j, 'loss', loss)
        return self



    def predict(self,X,use_correction):
        if not use_correction:
            return self.predict_predictive_fn(X.reshape((X.shape[0],1,X.shape[1],1)))
        else:
            alpha, beta, nu, mu  = self.predict_fn(X.reshape((X.shape[0],1,X.shape[1],1)))
            if(self.constant_alpha != 0):
                alpha = alpha*np.ones(beta.shape)
            var = calculateVariance(alpha,beta,nu)

            return mu, var

    def loss_value(self,X,y):
        mu, alpha, var = self.predict_predictive_fn(X.reshape((X.shape[0],1,X.shape[1],1)))
        loss = t.logpdf(y, df=alpha, loc=mu, scale=np.sqrt(var))
        loss = -loss

        return np.mean(loss)

    def get_all_parameters(self):
        values = []
        for v in self.params_c:
            values.append(v.get_value())

        return values