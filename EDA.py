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

import numpy as np

import matplotlib.pyplot as plt

def print_count_texts_of_DS (train_sourse, test_sourse):
    print('Количество обучающих текстов',len(train_sourse['data']))
    print('Количество тестовых текстов',len(test_sourse['data']))
    print('Количество уникальных меток', len(set(train_sourse['target'])))
    print()

    print('Количество обучающих меток',len(np.unique(train_sourse['target'])))
    print('Количество тестовых меток',len(np.unique(test_sourse['target'])))
    print()

    print('Пример входного набора:\n',train_sourse['data'][0].strip())
    print('Метка:\n',train_sourse['target'][0])


def print_texr_example(train_tokinized):
    print(' '.join(train_tokinized))


def print_unique_tokin (vocabulary):
    UNIQUE_WORDS_N = len(vocabulary)
    print('Количество уникальных токенов', UNIQUE_WORDS_N)
    print(list(vocabulary.keys())[:int(UNIQUE_WORDS_N/10)])

def show_hist_word_frequency_dist(word_doc_freq):
    plt.hist(word_doc_freq, bins=10)
    plt.title('Распределение относительных частот слов')
    plt.yscale('log')
    plt.show()

def show_hist_target_dist(train_sourse, test_sourse):
    UNIQUE_LABELS_N = len(set(train_sourse['target']))
    plt.hist([train_sourse['target'],test_sourse['target']], bins=np.arange(0, UNIQUE_LABELS_N+1), alpha=0.8)    
    plt.yscale('log')
    plt.xscale('linear')
    plt.title('Распределение меток в выборке')
    plt.show()

def spamming_of_text(test_sourse,UNIQUE_WORDS_N):
    countWordInCorpus = 0
    for tx in test_sourse['target']:
        countWordInCorpus += len(tx)
    print('Количество употреблённых слов: ', countWordInCorpus)
    print('В среднем, любое слово повторяется в корпусе: ', countWordInCorpus/UNIQUE_WORDS_N," раз")