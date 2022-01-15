import matplotlib.pyplot as plt
import numpy as np

# csv = np.loadtxt('result.csv', delimiter=',')
# plt.scatter(csv[2], csv[0])
# plt.scatter(csv[2], csv[1])
# plt.show()

csv = np.loadtxt('result_recursive.csv', delimiter=',')

plt.plot(csv[1], label="prediction: reservoir's y(t)", linewidth=1.5)
plt.plot(csv[2], label="answer: henon's x(t+1)", alpha=0.5, linewidth=1.5)
plt.ylim((-2, 2))
plt.legend(loc='upper right')
plt.title('henon y vs y_ans')
plt.show()

