import numpy as np
import matplotlib.pyplot as plt

football = np.random.randn(500) * 20 + 160
basketball = np.random.randn(500) * 10 + 190
heights = np.arange(90, 230, 10)


def random_classifier():
    return np.random.randint(2, size = 500)


def height_classifier(my_heights, height):
    res = []
    for h in my_heights:
        if h > height:
            res.append(1) # баскетболист
        else:
            res.append(0) # футболист

    return np.asarray(res)


def metrics(class0, class1):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(class0)):
        if class0[i] == 0:
            TP += 1
        elif class0[i] == 1:
            FN += 1

    for i in range(len(class1)):
        if class1[i] == 0:
            FP += 1
        elif class1[i] == 1:
            TN += 1
    return TP, FN, FP, TN


def precision_recall(TP, FN, FP, TN):
    if TP + FP == 0:
        precision = 0
        recall = 0
    elif TP + FN == 0:
        precision = 1
        recall = 1
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    accuracy = (TN + TP) / (TP + FP + FN + TN)
    return precision, recall, accuracy


# Классифицируем с двумя классификаторами по двум классам
class0_rand = random_classifier()
class1_rand = random_classifier()
class0_hc = height_classifier(football, 185)
class1_hc = height_classifier(basketball, 185)

# Вычисляем метрики
# Для случайного классификатора
TP1, FN1, FP1, TN1 = metrics(class0_rand, class1_rand)
precision1, recall1, accuracy1 = precision_recall(TP1, FN1, FP1, TN1)

# Для ростового классификатора с фиксированным height = 185
TP2, FN2, FP2, TN2 = metrics(class0_hc, class1_hc)
precision2, recall2, accuracy2 = precision_recall(TP2, FN2, FP2, TN2)
