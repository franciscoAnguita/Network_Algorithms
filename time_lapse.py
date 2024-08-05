import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



# # Initialize empty lists to store data
# iter_values = []
# time_lapse_values = []

# # Open the file and read line by line
# with open('all_nodes2', 'r') as file:
#     next(file)  # Skip the header
#     for line in file:
#         # Split the line into columns based on the tab delimiter
#         columns = line.strip().split('\t')
#         # Extract the values and append to the respective lists
#         iter_values.append(float(columns[0]))
#         time_lapse_values.append(float(columns[2]))

# # Convert lists to NumPy arrays
# iter = np.array(iter_values)
# time_lapse = np.array(time_lapse_values)

# # Read the file
# data = np.genfromtxt('all_nodes2', delimiter='\t', skip_header=True) #, dtype=float
# iter = data[:, 0]
# time_lapse = data[:, 2]


# Data
# 10 nodes
# iter = np.array([10, 20, 50, 100, 500, 1000, 2000, 4000])
# time_lapse = np.array([8.77455735206604, 10.14546823501587, 19.405954599380493, 32.007930517196655, 105.77629446983337, 252.07057857513428,
#                        391.41545581817627])

# 50 nodes
# iter = np.array([10, 20, 50, 100, 200, 400, 500])
# time_lapse = np.array([29.646284103393555, 105.28888821601868, 332.46614360809326, 705.5233690738678, 1087.3732209205627, 3102.1325187683105,
#                        2657.4157626628876])

# # 100 nodes
# iter = np.array([10, 20, 50, 100])
# time_lapse = np.array([57.0817494392395, 223.8973069190979, 1059.0460267066956, 2757.5756590366364])


# # # 1000 nodes
# iter = np.array([10, 20, 30])
# time_lapse = np.array([714.8869135379791, 2843.505400657654, 5667.473606586456])

# # iterations vs time
# iter = np.array([10,20,50,100,500,1000,20,20,20,20,30,20,20,10,20,50,100,200,400,10, 20,50,100,10,20,30])
# time_lapse = np.array([78.77455735206604,10.14546823501587,19.405954599380493,32.007930517196655,105.77629446983337,252.07057857513428,
# 40.344600439071655,44.16650652885437,70.9783182144165,58.04580593109131,112.63474798202515, 196.83703064918518,
# 196.65661120414734,29.646284103393555,105.28888821601868,332.46614360809326,705.5233690738678,1087.3732209205627,
# 3102.1325187683105,57.0817494392395,223.8973069190979,1059.0460267066956,2757.5756590366364,714.8869135379791,
# 2843.505400657654,5667.473606586456])


# # nodes vs time
# iter = np.array([10,10,10,10,10,10,20,20,20,20,20,40,40,50,50,50,50,50,50,100,100,100,100,1000,1000,1000])
# time_lapse = np.array([78.77455735206604,10.14546823501587,19.405954599380493,32.007930517196655,105.77629446983337,252.07057857513428,
# 40.344600439071655,44.16650652885437,70.9783182144165,58.04580593109131,112.63474798202515, 196.83703064918518,
# 196.65661120414734,29.646284103393555,105.28888821601868,332.46614360809326,705.5233690738678,1087.3732209205627,
# 3102.1325187683105,57.0817494392395,223.8973069190979,1059.0460267066956,2757.5756590366364,714.8869135379791,
# 2843.505400657654,5667.473606586456])


# 300 nodes
# iter = np.array([5,10,15,20,50,100,200,1000])
# time_lapse = np.array([872.23,1426.09,1606.01,1664.22,1672.34,1681.36,1690.80,1722.19])

# # 500 nodes
# iter = np.array([5,10])
# time_lapse = np.array([2383.10,3993.71])

# # # 600 nodes
# iter = np.array([5,10,15,20,25,30,35,40,45,50,100,1000])
# time_lapse = np.array([3484.38,5549.34,6138.96,6325.55,6377.55,6399.84,6412.28,6415.54,6417.61,6419.37,6428.47,6740.08])


# # 1000 nodes
# iter = np.array([5,10,15,20,25,30])
# time_lapse = np.array([9031.85,14252.05,16763.79,17246.36,17765.83,18073.45])

# iterations - time
iter = np.array([300,300,300,300,300,300,300,300,500,500,600,600,600,600,600,600,600,600,600,600,600,600,1000,1000,1000,1000,1000,1000])
time_lapse = np.array([872.23,1426.09,1606.01,1664.22,1672.34,1681.36,1690.80,1722.19,2383.10,3993.71,3484.38,
                       5549.34,6138.96,6325.55,6377.55,6399.84,6412.28,6415.54,6417.61,6419.37,6428.47,6740.08,
                       9031.85,14252.05,16763.79,17246.36,17765.83,18073.45])




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
extrapolated_iters = np.arange(10, 100, 10)  # Adjust the range as needed
extrapolated_time_lapse = linear_trend(extrapolated_iters)


# Plot
plt.scatter(iter, time_lapse)
# plt.plot(extrapolated_iters, extrapolated_time_lapse, label='Extrapolated Curve', color='red')
plt.xlabel('Nodes')
plt.ylabel('Time (seconds)')
plt.title('Time vs. Nodes - TOTAL')
plt.legend()
plt.grid(True)
plt.show()
