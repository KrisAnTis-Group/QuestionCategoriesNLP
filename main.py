
import data
from data import get_DataSet_on_numpy, tokenize_corpus, build_vocabulary

import EDA


#   импорт данных из дата сета *.csv
train_sourse = get_DataSet_on_numpy(subset = "train")
test_sourse = get_DataSet_on_numpy(subset = "test")

#    1. определим, какое колличество наборов мы имеем,
#    2. сравним количество обучающих и тестовых меток,
#     чтобы убедиться, что данные спарсились верно

EDA.print_count_texts_of_DS(train_sourse,test_sourse)


#   разбиение текстов на токены: буквы, цифры больше 4 символов
#   получаем две матрицы: строки(тексты) * столбцы (признаки)
train_tokinized = tokenize_corpus(train_sourse['data'])
test_tokinized = tokenize_corpus(test_sourse['data'])

#   выведем пример одного набора данных
EDA.print_texr_example(train_tokinized[0])

# строим словарь: слова -> цифры (нумеруем токены)
MAX_DF = 0.8
MIN_COUNT = 5
UNIQUE_LABELS_N = len(set(train_sourse['target']))
vocabulary, word_doc_freq = build_vocabulary (train_tokinized, max_doc_freq = MAX_DF, min_count = MIN_COUNT)

#   выведем количество уникальных токенов и меток
EDA.print_unique_tokin(vocabulary)
#   выведем Распределение относительных частот слов
EDA.show_hist_word_frequency_dist(word_doc_freq)
#   оценим распределение меток в обучающей и тестовой выборках
EDA.show_hist_target_dist(train_sourse,test_sourse)
#   оценим встречаемость слова в наборах (заспамленность текста)
EDA.spamming_of_text(train_sourse, len(vocabulary))