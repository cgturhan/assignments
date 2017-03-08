import numpy as np
from matplotlib import pyplot as plt

def f(x,w,b): 
    
    """
    Parameters
    ----------
    x is a N-by-D numpy array
    w is a D dimensional numpy array
    b is a scalar
    
    Return
    ------
    N dimensional numpy array
    """
    #x is converted to (N x (D + 1)) sized array called as x_biased
    #x_biased = np.zeros((x.shape[0],x.shape[1]+1))
    #x_biased[:,:-1] = x
    bias = np.empty((x.shape[0]))
    bias.fill(b)
    #x_biased[:,-1] = bias
    
    # vectorolized version of w'x + b 
    return sigmoid(np.dot(x, w) + b)



def sigmoid(t): # do not change this line!
    return 1.0 / (1.0 + np.exp(-t))
    # implement the sigmoid here
    # t is a N dimensional numpy array
    # Should return a N dimensional numpy array



def l2loss(x,y,w,b): # do not change this line!
    loss = np.sum(np.square(y-f(x,w,b)))
    
    # attemp vectorized gradient calculation 
    # z = f(x,y,w,b)
    # Lw = -2*x.T*(y-z)*(y-z**2)
    # Lb = -2*(y-z)*(y-z**2)
        
    # calculate gradient and update w,b parameters
    der_Lb = 0
    der_Lw = np.zeros(w.shape)
    for i in range(x.shape[0]):
            xi = x[i,:]
            yi = y[i]
            fi = f(xi,w,b)
            der_Lw = der_Lw  + (-2*xi*(yi-fi)*(fi-fi**2))
            der_Lb = der_Lb  + (-2*(yi-fi)*(fi-fi**2))
            
    return loss, der_Lw, der_Lb 
    # implement the l2loss here
    # x is a N-by-D numpy array
    # y is a N dimensional numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # Should return three items: (i) the L2 loss which is a scalar, (ii) the gradient with respect to w, (iii) the gradient with respect to b



def minimize_l2loss(x,y,w,b, num_iters=1000, eta=0.001): # do not change this line!
    losses = []
    
    for i in range(num_iters):
        loss, der_Lw, der_Lb = l2loss(x,y,w,b)
        losses.append(loss)     
        w = w - eta * der_Lw
        b = b - eta * der_Lb     
    
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    
    return w,b
    
    # implement the gradient descent here
    # x is a N-by-D numpy array
    # y is a N dimensional numpy array
    # w is a D dimensional numpy array
    # b is a scalar
    # num_iters (optional) is number of iterations
    # eta (optional) is the stepsize for the gradient descent
    # Should return the final values of w and b


def roc_curve(x,y):

    ROC = np.zeros((101,2))
    thr = np.linspace(0,1,101)
    i = 0;

    for t in thr:
        TP = np.logical_and(x >= t, y==1).sum()
        TN = np.logical_and(x < t, y==0).sum()
        FP = np.logical_and(x >= t, y==0).sum()
        FN = np.logical_and(x < t, y==1).sum()
        ROC[i,0] = float(FP / (FP + TN))
        ROC[i,1] = float (TP / (TP + FN))
        
        i = i + 1

    plt.plot(ROC[:,0], ROC[:,1], lw = 3)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.show()


def pr_curve(x,y):
    precision = np.zeros(101)
    recall = np.zeros(101)
    thr = np.linspace(0,1,101)
    i = 0;

    for t in thr:
        TP = np.logical_and(x >= t, y==1).sum()
        TN = np.logical_and(x < t, y==0).sum()
        FP = np.logical_and(x >= t, y==0).sum()
        FN = np.logical_and(x < t, y==1).sum()
        precision[i] = float(TP /  (TP + FP))
        recall[i] = float (TP /  (TP + FN))
        i = i + 1


    plt.plot(recall,precision, lw = 3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1,1.1])
    plt.grid()
    plt.show()

