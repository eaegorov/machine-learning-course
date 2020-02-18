import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(665)

eps = 0.001
eps0 = 0.0001
lambda_reg = 0.007
step = 0.01

points = 100
poly_deg = 15
x_train = np.linspace(0, 1, points)
gt_train = 30 * x_train * x_train
err = 2 * np.random.randn(points)
err[3] += 200
err[77] += 100
err[50] -= 100
#t_train = gt_train + err


# Делим данные на 3 части, три точки с сильно смещённой ошибкой - в обучающей
def generate_dataset(x_tr, gt, err):
    xx = x_tr
    yy = gt
    xx, yy = shuffle(xx, yy, random_state = 0)
    yy = yy + err

    dataset = {}
    dataset['x_train'] = xx[:80]
    dataset['y_train'] = yy[:80]

    dataset['x_valid'] = xx[80:90]
    dataset['y_valid'] = yy[80:90]

    dataset['x_test'] = xx[90:]
    dataset['y_test'] = yy[90:]
    return dataset

dataset = generate_dataset(x_train, gt_train, err)


def return_phi(X, n):
    phi_n = np.empty((len(X), n + 1))
    phi_n[:, 0] = 1
    phi_n[:, 1] = X
    for i in range(2, n + 1):
        phi_n[:, i] = phi_n[:, i - 1] * phi_n[:, 1]
    return phi_n


def loss(X, t, w, lamb, n):
    fi = return_phi(X, n)
    s = 0
    for i in range(len(X)):
        s += (t[i] - np.dot(w.reshape(1, 16), fi[i, :].reshape(16, 1))) ** 2
    s = s / 2 + (lamb / 2) * np.dot(w, w.T)
    s = float(s[0])
    return s

def gradient(X, t, w, lamb, n):
    fi = return_phi(X, n)
    s = [0] * len(w)
    for i in range(len(X)):
        s += np.dot(t[i] - np.dot(w.reshape(1, 16), fi[i, :].reshape(16, 1)),  fi[i, :].reshape(1, 16))
    s = -s + lamb * w
    return s


def gradient_descent(X, t, n, step, lamb):
    loss_vals = []
    w_next = np.random.rand(n + 1).reshape((1, n + 1)) / 100
    cant_stop = True
    eps = 1e-3
    while cant_stop:
        w_old = w_next
        w_next = w_old - step * gradient(X, t, w_old, lamb, n)
        temp = np.sqrt(np.sum((w_next - w_old) ** 2))
        loss_vals.append(loss(X, t, w_next, lamb, n))
        print(loss_vals[-1])
        if temp < eps:
            cant_stop = False
    return loss_vals, w_next


# loss_vals, w_itog = gradient_descent(x_train, t_train, poly_deg, step, lambda_reg)
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.plot(x_train, t_train, 'ro', markersize=3)
# ax1.plot(x_train, w_itog.dot(return_phi(x_train, poly_deg).T).flatten())
# ax2.plot(list(range(1, len(loss_vals) + 1)), loss_vals)
# ax2.set_xlabel("Итерация")
# ax2.set_ylabel("Ошибка")
# plt.show()

lambda_regs = [0, 5, 10, 20, 30, 40, 50, 60, 70]
train_errors = []
val_errors = []

# Подсчёт ошибок на тренировочной и валидационной выборках (валидация)
for lamb in lambda_regs:
    loss_vals, w_itog = gradient_descent(dataset['x_train'], dataset['y_train'], poly_deg, step, lamb)
    y_pred_train = w_itog.dot(return_phi(dataset['x_train'], poly_deg).T).flatten()
    MSE = np.mean((dataset['y_train'] - y_pred_train) ** 2)
    train_errors.append(round(MSE, 3))


    loss_vals, w_itog = gradient_descent(dataset['x_valid'], dataset['y_valid'], poly_deg, step, lamb)
    y_pred_val = w_itog.dot(return_phi(dataset['x_valid'], poly_deg).T).flatten()
    MSE = np.mean((dataset['y_valid'] - y_pred_val) ** 2)
    val_errors.append(round(MSE, 3))

# Лучшая модель на валидационной
the_best = np.argmin(val_errors)

# Подсчёт ошибки тестовой выборке для лучшей модели
def test_error(lambd):
    loss_vals, w_itog = gradient_descent(dataset['x_test'], dataset['y_test'], poly_deg, step, lambd)
    y_pred_test = w_itog.dot(return_phi(dataset['x_test'], poly_deg).T).flatten()
    MSE_test = np.mean((dataset['y_test'] - y_pred_test) ** 2)
    return round(MSE_test, 3)


test_err = test_error(lambda_regs[the_best])
print("Test Error on the best model: ", test_err)

# Построение графика-гистограммы для всех моделей с ощибками на обучающей и валидационной выборках
width = 0.3
confidences = train_errors
confidences1 = val_errors
plt.figure(figsize=(12, 8))
labels = [str(i) for i in lambda_regs]
bin_poses = np.arange(len(labels))
bin_poses1 = bin_poses + width
bins_art = plt.bar(bin_poses, confidences, width, label = "Training Error")
bins_art1 = plt.bar(bin_poses1, confidences1, width, label = "Validate Error")
plt.xlabel("Параметры lambda-регуляризации")
plt.ylabel("Error")
plt.title("График для всех моделей с ощибками на обучающей и валидационной выборках")
plt.xticks(bin_poses + width / 2, labels, size = 10)
plt.legend(loc=3)
for rect in bins_art:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.005, f"{height}", ha="center", va="bottom")
for rect in bins_art1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.005, f"{height}", ha="center", va="bottom")

plt.show()
