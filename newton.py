import numpy as np
import matplotlib.pyplot as plt

class Newton:
    ''' Finds coefficients of Newton's polynomial using memoization
    '''
    def __init__(self, x, y):
        self._memo = {}
        self._x = x
        self._y = y

    ''' Provide a tuple of (highest power, 0) will calculate and store the
        divided differences
    '''
    def calculate(self, f_k2_k1):
        assert(len(f_k2_k1) == 2)
        if f_k2_k1 in self._memo:
            return self._memo[f_k2_k1]

        if f_k2_k1[0] == f_k2_k1[1]:
            self._memo[f_k2_k1] = self._y[f_k2_k1[0]]
            return self._memo[f_k2_k1]

        start = f_k2_k1[0]
        end = f_k2_k1[1]
        start_minus_1 = start - 1
        end_plus_1 = end + 1
        f_b = (start, end + 1)
        f_a = (start - 1, end)

        self._memo[f_k2_k1] = (self.calculate(f_b) - self.calculate(f_a)) / (self._x[start] - self._x[end])
        return self._memo[f_k2_k1]
            
    ''' Retrieve the coefficients 
    '''
    def coefficients(self):
        coef = []
        k = len(self._x)
        for i in range(1, k+1):
            lookup = (k - i, 0)
            coef.append(self._memo[lookup])
        return coef

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
    n = len(coefficients)
    p = coefficients[n-1]
    for i in range(0, n):
        j = n - 2 - i
        p = coefficients[j] + (x - data_x[j]) * p
    return p

def main():
    # number of data points (also degree of polynomial + 1)
    num_points = 4

    # generate random points around line x = y
    data_x, data_y = generate_random_points(num_points)

    # use Newton interpolation to calculate coefficients
    newton = Newton(data_x, data_y)
    f = (num_points-1, 0)
    newton.calculate(f)
    coefficients = newton.coefficients()

    # create the range for the polynomial's plot
    step = 0.0001
    start = 10
    end = start + num_points + step - 1
    x = np.arange(start, end, step)

    # plot the polynomial given the coefficients
    plt.plot(x, newton_polynomial(x, data_x, coefficients), color='r')

    # scatter plot the generated data points
    plt.scatter(data_x, data_y, color='b')

    # show the plots
    plt.show()
    

if __name__ == '__main__':
    main()

