import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(num_points):
    x = []
    y = []
    for i in range(num_points):
        x_i = i + 10
        y_i = x_i + np.random.uniform(-1, 1)
        x.append(x_i)
        y.append(y_i)
    return x, y

def calculate_coefficients(x, y):
    assert(len(x) == len(y))
    
    return [0, 1]

def polynomial(x, coefficients):
    y = 0
    for i in range(len(coefficients)):
        y += coefficients[i] * x**i
    return y

def main():
    # generate random points around line x = y
    data_x, data_y = generate_random_points(21)

    # use Newton interpolation to calculate coefficients
    coefficients = calculate_coefficients(data_x, data_y)

    # create the range for the polynomial's plot
    start = 0
    end = 32
    step = 0.01
    x = np.arange(start, end, step)

    # plot the polynomial given the coefficients
    y = polynomial(x, coefficients)
    plt.plot(x, y, color='r')

    # scatter plot the generated data points
    plt.scatter(data_x, data_y, color='b')

    # show the plots
    plt.show()
    

if __name__ == '__main__':
    main()

