import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from itertools import combinations

# Data and ground truth
x = np.linspace(0.1, 2 * np.pi, 1000)
gt = 100 * np.sin(x) + 0.5 * np.exp(x) + 300
eps = 10 * np.random.randn(1000)
data = gt + eps


# Splitting the data (in other case may use sklearn - train_test_split)
def generate_dataset(x, data):
    xx = x
    yy = data
    xx, yy = shuffle(xx, yy, random_state = 0)

    dataset = {}
    dataset['x_train'] = xx[:800]
    dataset['y_train'] = yy[:800]

    dataset['x_valid'] = xx[800:900]
    dataset['y_valid'] = yy[800:900]

    dataset['x_test'] = xx[900:]
    dataset['y_test'] = yy[900:]
    return dataset


dataset = generate_dataset(x, data)

def x_1(a):   # x в первой стенени
    b = np.array(a)
    return b


def x_2(a):  # x в квадрате
    b = np.array(a) ** 2
    return b


def x_3(a):  # x в кубе
    b = np.array(a) ** 3
    return b


# Basic functions
basics_name = ['sin(x)', 'cos(x)', 'exp(x)', 'ln(x)', 'x', 'x^2', 'x^3', 'sqrt(x)']
basic_functions = [np.sin, np.cos, np.exp, np.log, x_1, x_2, x_3, np.sqrt]

error_list = []
train_e = []  # для диаграммы, ошибки на обучающей выборке
weights = []
combins1 = []  # для строковых названий функций
combins2 = []
for i in range (1, 4):
    comb = combinations(basic_functions, i)
    comb2 = combinations(basics_name, i)
    for Touple in comb:
        lines = [np.ones(dataset['x_train'].shape[0])] + list(map(lambda k: Touple[k](dataset['x_train']), range(0, len(Touple))))
        X = np.array(lines).T
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), dataset['y_train'])
        y_pred_train = np.dot(X, w)
        MSE = np.mean((dataset['y_train'] - y_pred_train) ** 2)
        train_e.append(MSE)

        lines_val = [np.ones(dataset['x_valid'].shape[0])] + list(map(lambda k: Touple[k](dataset['x_valid']), range(0, len(Touple))))
        X_1 = np.array(lines_val).T
        y_pred_val = np.dot(X_1, w)
        MSE = np.mean((dataset['y_valid'] - y_pred_val) ** 2)

        weights.append(w)
        error_list.append(MSE)
        combins2.append(Touple)

    for Touple in comb2:
        combins1.append(Touple)


# поиск индексов трёх лучших моделей на валидационной выборке
def bestmodels (error_list):
    a = error_list
    min1 = a[0]
    min2 = a[0]
    min3 = a[0]
    for elem in a:
        if elem < min1:
            min3 = min2
            min2 = min1
            min1 = elem
        elif elem < min2:
            min3 = min2
            min2 = elem
        elif elem < min3:
            min3 = elem

    idx1 = a.index(min1)
    idx2 = a.index(min2)
    idx3 = a.index(min3)
    ilist = [idx1, idx2, idx3]
    return ilist


# списки для вычисления лучшей модели из трёх
bests_indexes = bestmodels(error_list)
best_functions = [combins2[i] for i in bests_indexes]
best_weights = [weights[i] for i in bests_indexes]

# процесс подготовки и рисования диаграммы для 3-х лучших моделей
names = [combins1[i] for i in bests_indexes]  # строковые комбинации функций 3-х лучших моделей
def make_model (names, best_weights, n):  # составление строковой модели для диаграммы
    model = str(round(best_weights[n - 1][0], 2))
    j = 1
    for i in range(0, len(names[n - 1])):
        model += ' + ' + str(round(best_weights[n - 1][j], 2)) + ' * ' + str(names[n - 1][i])
        j += 1

    return model


model1 = make_model(names, best_weights, 1)
model2 = make_model(names, best_weights, 2)
model3 = make_model(names, best_weights, 3)

train_err = [round(train_e[i], 3) for i in bests_indexes]
valid_err = [round(error_list[i], 3) for i in  bests_indexes]

width = 0.3
confidences = train_err
confidences1 = valid_err
plt.figure(figsize=(12, 8))
labels = [model1, model2, model3]
bin_poses = np.arange(len(labels))
bin_poses1 = bin_poses + width
bins_art = plt.bar(bin_poses, confidences, width, label = "Training Error")
bins_art1 = plt.bar(bin_poses1, confidences1, width, label = "Validating Error")
plt.xlabel("Модели регрессии")
plt.ylabel("Error")
plt.title("Диаграмма 3-х лучших моделей")
plt.xticks(bin_poses + width / 2, labels, size = 8)
plt.legend(loc = 3)
for rect in bins_art:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.005, f"{height}", ha="center", va="bottom")
for rect in bins_art1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height() * 1.005, f"{height}", ha="center", va="bottom")


# выявление лучшей модели на тестовой выборке
best_error = []
t = 0
for Touple in best_functions:
    lines_test = [np.ones(dataset['x_test'].shape[0])] + list(map(lambda k: Touple[k](dataset['x_test']), range(0, len(Touple))))
    X_2 = np.array(lines_test).T
    y_pred = np.dot(X_2, best_weights[t])
    MSE = np.mean((dataset['y_test'] - y_pred) ** 2)
    best_error.append(MSE)
    t += 1

best_i = best_error.index(min(best_error))
print('The best model on Test Set is: ', labels[best_i])
print('Error (MSE): ', round(best_error[best_i], 3))

# #plt.savefig("best_models.png")
plt.show()