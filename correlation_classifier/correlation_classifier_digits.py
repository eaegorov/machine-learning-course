from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#Загрузка датасета
digits = datasets.load_digits()


# Показать случайные картинки
print(digits.data.shape)
print(digits.target)
fig, axes = plt.subplots(4,4)
axes = axes.flatten()
for i, ax in enumerate(axes):
    dig_ind=np.random.randint(0, len(digits.images))
    ax.imshow(digits.images[dig_ind].reshape(8, 8))
    ax.set_title(digits.target[dig_ind])
plt.show()


#Посчитать картинок какого класса сколько
dic = {x:0 for x in range(10)}
for dig in digits.target:
    dic[dig] += 1
# print(dic)


def prepare_data(data, avg):
    """
    Подготавливает данные для кореляционного классификатора
    :param data: np.array, данные (размер выборки, количество пикселей
    :return: data: np.array, данные (размер выборки, количество пикселей
    """
    return data - avg

def train_val_test_split(data, labels):
    """
    Делит выборку на обучающий и тестовый датасет
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.30)
    validation_data, test_data, validation_labels, test_labels = train_test_split(test_data, test_labels, test_size = 0.5)

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def softmax(vec):
    a = np.argmax(vec)

    sftmx = []
    for i in range(len(vec)):
        sftmx.append((np.exp(vec[i] - vec[a])) / np.sum(np.exp(vec - vec[a])))

    return sftmx


class CorrelationClassifier:
    def __init__(self, classes_count=10):
        self.classes_count=classes_count

    def fit(self, data, labels):
        """
        Производит обучение алгоритма на заданном датасете
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        self.averages = [[0 for i in range(data.shape[1])] for j in range(10)]

        for i in range(data.shape[0]):
            self.averages[labels[i]] += data[i, :]

        self.averages = np.array(self.averages) / dic[i]

    def predict(self, data):
        """
        Предсказывает вектор вероятностей для каждого наблюдения в выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :return: np.array, результаты (len(data), count_of_classes)
        """

        res = self.averages * data
        res = [np.sum(res[i]) for i in range(10)]
        probability = softmax(res)
        return probability

    def accuracy(self, data, labels):
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        true_ans = 0
        for i in range(data.shape[0]):
            p = self.predict(data[i, :])
            if np.argmax(p) == labels[i]:
                true_ans += 1

        return true_ans / data.shape[0]

    # Confusion Matrix
    def conf_matrix(self, data, labels):
        size = (10, 10)
        conf = np.zeros(size, dtype=int)

        for i in range(data.shape[0]):
            p = self.predict(data[i, :])
            conf[labels[i], np.argmax(p)] += 1

        return conf


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(digits.data, digits.target)

train_data = prepare_data(train_data, 8)
validation_data = prepare_data(validation_data, 8)
test_data = prepare_data(test_data, 8)

# Посчитать картинок какого класса сколько в обучающем датасете
# dic={x:0 for x in range(10)}
# for dig in train_labels:
#     dic[dig]+=1
# print(dic)

classifier = CorelationClassifier()
classifier.fit(train_data, train_labels)
print(f"Training accuracy {classifier.accuracy(train_data, train_labels)}")
print(f"Validation accuracy {classifier.accuracy(validation_data, validation_labels)}")


# Вывод Confusion Matrix
print("Confusion Matrix for the TEST SET: ")
print(classifier.conf_matrix(test_data, test_labels))

CM = classifier.conf_matrix(test_data, test_labels)


# Precison and Recall для всех классов в Confusion Matrix
def precision_recall(matrix):
    thresholds = np.linspace(0.5, 1, 11)
    precision = []
    recall = []

    for i in range(10):
        TP = np.sum(matrix[i, i])
        FN = np.sum(matrix[i, :])
        FP = np.sum(matrix[:, i])
        precision.append(round((TP / (TP + FP)), 2))
        recall.append(round((TP / (TP + FN)), 2))

    return precision, recall


precision, recall = precision_recall(CM)
print("Precisions: ", precision)
print("Recalls: ", recall)
