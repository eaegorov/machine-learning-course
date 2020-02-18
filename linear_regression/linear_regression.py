import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 1000)
gt = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
eps = 10 * np.random.randn(1000)
data = gt + eps

degree_list = list(range(1, 21))

fig, axes = plt.subplots(4, 5, constrained_layout = True, figsize = (15, 10))

error_list = []

row = 0
col = 0
# 20 polynomials
for degree in degree_list:
    lines = [np.ones(x.shape[0])] + list(map(lambda n: x**n, range(1, degree + 1)))
    X = np.array(lines).T
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), data)
    y_pred = np.dot(X, w)
    SSE = np.sum((data - y_pred) ** 2)
    error_list.append(SSE)
    axes[row, col].plot(x, y_pred, 'r')
    axes[row, col].plot(x, gt, 'b--')
    axes[row, col].set_title('degree = ' + str(degree), pad = 3)
    col += 1
    if col > 4:
        col = 0
        row += 1


#plt.savefig("20_polynomials.png")
plt.show()
