import theano.tensor as T

def psi(x):

    #shift to get good numerical approximation and then use recurrence formula

    evaluate_x = x + 6.0
    inv_evaluate_x = 1.0/evaluate_x
    inv_evaluate_x_sqr = inv_evaluate_x**2
    psi = T.log(evaluate_x) - inv_evaluate_x/2.0+inv_evaluate_x_sqr*(-1.0/12.0+inv_evaluate_x_sqr*(1.0/120.0+inv_evaluate_x*(1.0/252.0 + inv_evaluate_x_sqr*(1.0/240.0-5.0/660.0*inv_evaluate_x_sqr))))

    psi = psi - 1.0/(x+5.0)
    psi = psi - 1.0/(x+4.0)
    psi = psi - 1.0/(x+3.0)
    psi = psi - 1.0/(x+2.0)
    psi = psi - 1.0/(x+1.0)
    psi = psi - 1.0/x
    return psi


