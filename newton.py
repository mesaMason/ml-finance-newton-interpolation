import argparse
import numpy as np
import matplotlib.pyplot as plt

class Newton:
    ''' Finds coefficients of Newton's polynomial using memoization
    '''

    def __init__(self, x, y):
        self._memo = {}
        self._x = x
        self._y = y

    def calculate(self, f_k_j):
        ''' Divided difference f[x_k, ... , x_j] is represented by the tuple (x_k, x_j)
        '''
        assert(len(f_k_j) == 2)
        if f_k_j in self._memo:
            return self._memo[f_k_j]

        # base case of f(x_n) = y_n, represented by the tuple (x_n, x_n)
        if f_k_j[0] == f_k_j[1]:
            self._memo[f_k_j] = self._y[f_k_j[0]]
            return self._memo[f_k_j]

        start = f_k_j[0]
        end = f_k_j[1]
        start_minus_1 = start - 1
        end_plus_1 = end + 1

        f_b = (start, end + 1) # f[x_k, ... , x_j+1]
        f_a = (start - 1, end) # f[x_k-1, ... , x_j]

        self._memo[f_k_j] = \
            (self.calculate(f_b) - self.calculate(f_a)) / (self._x[start] - self._x[end])

        return self._memo[f_k_j]
            
    def coefficients(self):
        ''' Retrieve the coefficients 
        '''
        coef = []
        k = len(self._x)
        for i in range(0, k):
            lookup = (i, 0)
            coef.append(self._memo[lookup])
        return coef

def parse_args():
    parser = argparse.ArgumentParser(description='Regression')
    parser.add_argument("-n",
                        "--num_points",
                        required=True,
                        type=int,
                        help='Number of test data points to generate')
    args = parser.parse_args()
    if args.num_points < 1:
        raise Exception("num_points must be greater than 0")
    return args

def generate_random_points(num_points):
    x = []
    y = []
    for i in range(num_points):
        x_i = i + 10
        y_i = x_i + np.random.uniform(-1, 1)
        x.append(x_i)
        y.append(y_i)
    return x, y

def newton_polynomial(x, data_x, coefficients):
    ''' Evaluate the Newton polynomial using nested multiplication per:
        https://pages.cs.wisc.edu/~amos/412/lecture-notes/lecture08.pdf
    '''
    assert(len(data_x) == len(coefficients))
    n = len(coefficients) - 1
    p = coefficients[n]
    for i in range(0, n):
        j = n - i - 1
        p = coefficients[j] + (x - data_x[j]) * p
    return p

def main():
    args = parse_args()
    num_points = args.num_points

    # generate random points around line x = y
    data_x, data_y = generate_random_points(num_points)

    # use Newton interpolation to calculate coefficients
    newton = Newton(data_x, data_y)
    f = (num_points-1, 0)
    newton.calculate(f)
    coefficients = newton.coefficients()

    # create the range for the polynomial's plot
    step = 0.01
    start = 10 - 1/num_points
    end = 10 + num_points - 1 + 1/num_points

    # evaluate and plot the polynomial given the coefficients
    x = np.arange(start, end, step)
    y = newton_polynomial(x, data_x, coefficients)
    plt.plot(x, y, color='r')

    # scatter plot the generated data points
    plt.scatter(data_x, data_y, color='b')

    # show the plots
    plt.show()

if __name__ == '__main__':
    main()
