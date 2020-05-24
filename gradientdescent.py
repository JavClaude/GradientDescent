import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def f(x, y):
    '''
    Function to optimize by GradientDescent
    '''
    return x**2 + np.sin(y**2)

def d_f(x, y):
    '''
    Derivative of f(x)

    g(f(x))' = g'(f(x))*f'(x)
    '''
    g1 = 2*x
    g2 = np.cos(y**2)*2*y
    return np.array([g1, g2])

class GradientDescent(object):
    def __init__(self,
                f,
                d_f,
                learning_rate,
                maxiter,
                min_error):
        
        self.f = f
        self.d_f = d_f
        self.learning_rate = learning_rate

        self.maxiter = maxiter
        self.min_error = min_error
        self.it = 0

        self.iter_x, self.iter_y, self.iter_count = np.empty(0), np.empty(0), np.empty(0)
        self.gradients = np.array([[0, 0]])

    def step(self, x_init: int = 3, y_init: int = 2):
        self.error = 10
        x, y = x_init, y_init
        X = np.array([x_init, y_init])

        while np.linalg.norm(self.error) > self.min_error and self.it < self.maxiter:
            self.it += 1
            self.iter_x = np.append(self.iter_x, x)
            self.iter_y = np.append(self.iter_y, y)
            self.iter_count = np.append(self.iter_count, self.it)

            X_prev = X
            grads = self.d_f(x, y)

            self.gradients = np.append(self.gradients, [grads], 0)

            X = X - self.learning_rate * grads
            x, y = X[0], X[1]

            self.error = X - X_prev
            
            logger.info("Iteration: {}\nGradients: {}: \nError: {}\n ---------".format(self.it, grads, self.error))
        
        logger.info("Stop Descent at:\nx: {}\ny:{}".format(x, y))
        self.__fitted = True
    
    def clear(self):
        self.error = 10
        self.it = 0
        self.iter_x, self.iter_y = 0, 0
        self.gradients = np.array([[0, 0]])
        self.iter_count = 0

    def plot_descent(self, x_min: int = -4, x_max: int = 4, y_min: int = -4, y_max: int = 4, nb_points: int = 200, save=False):
        if self.__fitted:
            x = np.linspace(x_min, x_max, nb_points)
            y = np.linspace(y_min, y_max, nb_points)

            X, Y = np.meshgrid(x, y)
            Z = self.f(X, Y)
            
            ## plot ##
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap="jet", alpha=0.4)
            ax.plot(self.iter_x, self.iter_y, self.f(self.iter_x, self.iter_y), color='r', marker="*", alpha=0.4)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
            if save:
                plt.savefig("GradientDescent.jpg")

        else:
            raise ValueError("Please call the step function before")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", help="LR to use for gradient descent", type=float, default=0.01)
    parser.add_argument("--max_iter", help="Maximum number of iteration", type=int, default=200)
    parser.add_argument("--min_error", help="Min error acceptable until convergence", type=float, default=0.01)
    parser.add_argument("--x_init", help="Value of x", type=int, default=4)
    parser.add_argument("--y_init", help="Value of y", type=int, default=2)

    args = parser.parse_args()

    logger.info("Run Gradient Descent with:\nLR: {}\nmax_iter: {}\nmin_grad_error: {}".format(args.lr, args.max_iter, args.min_error))
    Optim = GradientDescent(f, d_f, args.lr, args.max_iter, args.min_error)
    Optim.step(args.x_init, args.y_init)
    Optim.plot_descent()

if __name__ == "__main__":
    main()