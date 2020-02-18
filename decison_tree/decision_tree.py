import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split


def pickle_it(data, path):
    """
    Сохранить данные data в файл path
    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_it(path):
    """
    Достать данные из pickle файла
    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


class TreeNode:
    divide_value = None
    split_characteristic = None

    left_child = None
    right_child = None

    is_leaf = False
    labels = None
    classes_count = None

    def __init__(self, divide_value, split_characteristic, left_child, right_child,
                 is_leaf=False, labels=None, classes_count=None):
        self.left_child = left_child
        self.right_child = right_child
        self.divide_value = divide_value
        self.split_characteristic = split_characteristic

    def isLeaf(self, labels):
        probs = []
        class_counts = DeсisionTree.get_classes(labels)
        num_elements = 0
        for i in range(10):
            num_elements += class_counts[i]
        for i in range(10):
            probs.append(class_counts[i] / num_elements)

        return probs


class DeсisionTree:
    MAX_HEIGHT_TREE = None
    MIN_ENTROPY = None
    STEPS_COUNT = None
    FEATURES_SIZE = None
    FEATURES_TO_EVALUATE = None
    MIN_NODE_ELEMENTS = None
    CLASSES_COUNT=None

    data = None
    labels = None
    root = None

    def __init__(self, max_height_tree, min_entropy, steps_count, min_node_elements, times_to_evaluate):
        self.MAX_HEIGHT_TREE = max_height_tree
        self.MIN_ENTROPY = min_entropy
        self.STEPS_COUNT = steps_count
        self.MIN_NODE_ELEMENTS = min_node_elements
        self.TIMES_TO_EVALUATE = times_to_evaluate

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.VECTOR_SIZE = data.shape[1]
        self.CLASSES_COUNT = labels.shape[1]

        self.root = self.slowbuildTree(self.data, self.labels, 0)

    def getEntropy(self, labels):
        # Оценивание энтропии по labels
        class_counts = self.get_classes(labels)
        S_i = 0
        for i in range(10):
            S_i += class_counts[i]

        H = 0
        for k in range(10):
            H += (class_counts[k] / S_i) * np.log2(class_counts[k])

        return H - np.log2(S_i)

    def get_classes(self, labels):
        # Гистограмма классов, содержащихся в labels
        dic = {x:0 for x in range(10)}
        for l in labels:
            dic[l] += 1
        return dic

    def partition(self, data, labels, x_i, mu):
        true_digs = data[:, x_i] >= mu
        true_digs = data[true_digs]

        false_digs = data[:, x_i] < mu
        false_digs = data[false_digs]

        return true_digs, false_digs

    def find_best_split(self, data, labels):
        info_gain = []
        mu = []
        for i in range(self.STEPS_COUNT):
            x_i = random.randint(0, self.VECTOR_SIZE - 1)
            mu_i = random.randint(data.min(), data.max())
            m = data[:, x_i] >= mu_i
            entropy = self.getEntropy(labels[m])
            ig = entropy - np.sum()
            info_gain.append(ig)
            mu.append(mu_i)

        best_ig = info_gain[np.argmax(info_gain)]
        best_mu = mu[np.argmax(mu)]
        return best_ig, best_mu

    def slowBuildTree(self, data, labels, height_tree):

        print("\tHeight tree %s, length data %s" % (height_tree, len(data)))

        allDataEntropy = self.getEntropy(labels)

        """
        Условия на создание терминального узла
        Вывод информации о каждом случае в консоль
        """
        if allDataEntropy < self.MIN_ENTROPY:
            print("Энтропия в узле меньше определённого порога")
            return TreeNode(isLeaf=True, labels=labels).isLeaf(labels)

        elif height_tree == self.MAX_HEIGHT_TREE:
            print("Глубина дерева дистигла максимального значения")
            return TreeNode(isLeaf=True, labels=labels).isLeaf(labels)

        elif data < self.MIN_NODE_ELEMENTS:
            print("Количество элементов обучающей выборки, достигших узла меньше определенного порога")
            return TreeNode(isLeaf=True, labels=labels).isLeaf(labels)

        """
        Подсчёт лучшего разделения и поиск лучшего information gain
        """
        info, mu = self.find_best_split(data, labels)

        """
        Деление выборки и перенаправление в новые узлы, рекурсивный вызов этой функции
        """
        true_digs, false_digs = self.partition(data, labels)
        left_node = self.slowBuildTree(true_digs)
        right_node = self.slowBuildTree(false_digs)

        return tree_node


    def getPrediction(self, data):
        #Возвращает предсказание для новых данных на основе корня дерева
        pass


class RandomForest:
    # TODO
    pass


def generate_dataset(data_dig, target_dig):
    dataset = {}
    dataset['x_train'], dataset['x_test'], dataset['y_train'], dataset['y_test'] = train_test_split(data_dig, target_dig, test_size = 0.30)
    dataset['x_valid'], dataset['x_test'], dataset['y_valid'], dataset['y_test'] = train_test_split(dataset['x_test'], dataset['y_test'], test_size = 0.5)

    return dataset


if __name__=="__main__":

    """
    Загрузка датасета digits
    """
    digits = datasets.load_digits()
    data = digits.data
    labels = digits.target

    """
    Формирование выборки
    """
    dataset = generate_dataset(data, labels)

    """
    Валидация по количеству случайных семплирований 5, 50, 250, 500, 1000, +500 в зависимости от мощности компьютера
    """
    acc = []
    s = [5, 50, 250, 500, 1000]
    for i in s:
        tree = DeсisionTree(100, 0.1, 2, 1, steps_count=s)
        tree.train(data, labels)
        good_predictions = 0
        bad_predictions = 0
        for i, element in enumerate(data):
            probability, prCl = getPrediction(element, root)
            if prCl == real_classes[i]:
                good_predictions += 1
            else:
                bad_predictions += 1
        accuracy = (float(good_predictions)/(good_predictions + bad_predictions))
        acc.append(accuracy)
    print("Best steps_count is: ", s[np.argmax(accuracy)])





