import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
# iter = np.array([10, 20, 50, 100, 500, 1000])
# time_lapse = np.array([8.77455735206604, 10.14546823501587, 19.405954599380493, 32.007930517196655, 105.77629446983337, 252.07057857513428])

# 10 nodes
iter = np.array([10, 20, 50, 100, 500, 1000, 2000])
time_lapse = np.array([8.77455735206604, 10.14546823501587, 19.405954599380493, 32.007930517196655, 105.77629446983337, 252.07057857513428,
                       391.41545581817627])



# Define the function to fit (you can choose a different function based on your data)
# def func(x, a, b):
#     return a * np.exp(b * x)

# # Fit the curve
# popt, pcov = curve_fit(func, iter, time_lapse)

# # Extrapolate for more iterations
# extrapolated_iters = np.arange(10, 1000, 10) # Adjust the range as needed
# extrapolated_time_lapse = func(extrapolated_iters, *popt)

# Fit linear regression
coefficients = np.polyfit(iter, time_lapse, 1)
linear_trend = np.poly1d(coefficients)

# Extrapolate for more iterations
extrapolated_iters = np.arange(10, 1000, 10)  # Adjust the range as needed
extrapolated_time_lapse = linear_trend(extrapolated_iters)


# Plot
"""  plt.scatter(iter, time_lapse, label='Data')
plt.plot(extrapolated_iters, extrapolated_time_lapse, label='Extrapolated Curve', color='red')
plt.xlabel('Iterations')
plt.ylabel('Time (seconds)')
plt.title('Time vs. Iterations')
plt.legend()
plt.grid(True)
plt.show() """
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def polynomial_adjustment(X, Y, degree):
    # Fit the polynomial of given degree
    coefficients = np.polyfit(X, Y, degree)
    
    # Generate points for the polynomial curve
    X_curve = np.linspace(min(X), max(X), 100)
    Y_curve = np.polyval(coefficients, X_curve)
    
    return X_curve, Y_curve

degrees = [1,2,3,4]

# Set the limits for x-axis and y-axis
plt.xlim(-100, 2000)  # Adjust the limits as needed for the x-axis
plt.ylim(0, max(time_lapse) + 5000)  # Adjust the limits as needed for the y-axis

X = 2 * np.random.rand(100, 1)
Y = 5 + 3 * X + np.random.randn(100, 1)
plt.scatter(iter, time_lapse, label='Data')

rmses = [] # Lista, en la que tendremos en orden creciente de orden polinomial los errores (inversa de precision) de nuestros modelos
for degree in degrees:
    X_curve, Y_curve = polynomial_adjustment(iter, time_lapse, degree)
    coefficients = np.polyfit(iter, time_lapse, degree)
    prediction = np.polyval(coefficients, iter)
    error = rmse(time_lapse, prediction)
    rmses.append(error) 
    plt.plot(X_curve, Y_curve, label='Degree {}'.format(degree))

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Adjustment with Variable Degree')
plt.legend()
plt.show()

plt.plot(degrees, rmses)

plt.xlabel('Degree')
plt.ylabel('RMSE')
plt.title('Polynomial Adjustment RMSE by Degree')
plt.legend()
plt.show()
