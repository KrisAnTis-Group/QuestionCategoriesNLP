# на этапе предварительного развет. анализа было решено проанализировать данные по следующим критериям:
# 1. определить количество обучающих текстов и тестовых, вывести пример обрабатываемого данного.
# 2. построить гистограмму частотности слов, обратить внимание на распределение гистограммы
# 3. определить количество уникальных токенов
# 4. отобразить 10% самых частотных слов
# 5. оценить процент заполненности матрицы признаков для обучающей и тестовой выборки
# 6. построить гистограмму распределения весов признаков
# 7. определить колличество уникальных классов (меток)
# 8. построить гистограмму распределения меток в обучающей и тестовой выборках
# 9. оценить отношение размера словаря к размеру текста
# 10. оценить процентное соотношение ненормативной лексики в словаре к размеру словаря
# 11. определить тональности текстов по оценки агрессивности

import numpy as np

import matplotlib.pyplot as plt

import data
from data import get_DataSet_on_numpy, tokenize_corpus, build_vocabulary

# импорт данных из дата сета *.csv
train_sourse = get_DataSet_on_numpy(subset = "train")
test_sourse = get_DataSet_on_numpy(subset = "test")

print('Количество обучающих текстов',len(train_sourse['data']))
print('Количество тестовых текстов',len(test_sourse['data']))
print()
print(train_sourse['data'][0].strip())
print()
print('Метка: ',train_sourse['target'][0])

print('Количество обучающих меток',len(np.unique(train_sourse['target'])))
print('Количество тестовых меток',len(np.unique(test_sourse['target'])))

# разбиение текстов на токены: буквы, цифры больше 4 символов
# получаем две матрицы: строки(тексты) * столбцы (признаки)
train_tokinized = tokenize_corpus(train_sourse['data'])
test_tokinized = tokenize_corpus(test_sourse['data'])

print(' '.join(train_tokinized[0]))

# строим словарь: слова -> цифры (нумеруем токены)
MAX_DF = 0.8
MIN_COUNT = 5

vocabulary, word_doc_freq = build_vocabulary (train_tokinized, max_doc_freq = MAX_DF, min_count = MIN_COUNT)

UNIQUE_WORDS_N = len(vocabulary)
print('Количество уникальных токенов', UNIQUE_WORDS_N)
print(list(vocabulary.items())[:int(UNIQUE_WORDS_N/10)])

plt.hist(word_doc_freq, bins=10)
plt.title('Распределение относительных частот слов')
plt.yscale('log')
plt.show()

UNIQUE_LABELS_N = len(set(train_sourse['target']))
print('Количество уникальных меток', UNIQUE_LABELS_N)


plt.hist(train_sourse['target'], bins=np.arange(0, UNIQUE_LABELS_N+1))
plt.yscale('log')
plt.xscale('linear')
plt.title('Распределение меток в обучающей выборке')
plt.show()

plt.hist(test_sourse['target'], bins=np.arange(0, UNIQUE_LABELS_N+1))
plt.title('Распределение меток в тестовой выборке')
plt.yscale('log')
plt.xscale('linear')
plt.show()