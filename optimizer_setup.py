import lasagne

def create_updates(configuration, loss, params):
    if configuration['name'] == 'adam':
        updates = lasagne.updates.adam(loss, params, configuration['learning_rate'], configuration['beta1'], configuration['beta2'])
    elif configuration['name'] == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, configuration['learning_rate'], configuration['rho'])
    elif configuration['name'] == 'nesterov_momentum':
        updates = lasagne.updates.nesterov_momentum(loss, params, configuration['learning_rate'], configuration['momentum'])

    return updates