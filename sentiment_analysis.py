from __future__ import division
from collections import defaultdict
from math import log
import operator
import re
import random


with open('dictionary.txt', 'r') as data_set:
    data_set = data_set.read().split('\n')

with open('sentiment_labels.txt', 'r') as labels:
    labels = labels.read().split('\n')

temp = zip(data_set, labels)
random.shuffle(temp)

data_set, labels = [], []

for __ in temp:
    data_set.append(__[0])
    labels.append(__[1])

parsed_data = dict()
parsed_label = dict()
data_count = len(data_set)

number_of_train_data = int(0.85*data_count)

VN, N, NN, P, VP = 'vn', 'n', 'nn', 'p', 'vp'
labels_dict = {
    N: 1,
    NN: 2,
    P: 3,
}

likelihood = defaultdict(dict)

for data_raw, label_raw in zip(data_set, labels):
    data, phrase_id = data_raw.split('|')
    phrase_id_, score = label_raw.split('|')

    # print phrase_id, phrase_id_
    phrase_id, phrase_id_ = int(phrase_id), int(phrase_id_)
    score = float(score)

    parsed_data[phrase_id] = data
    parsed_label[phrase_id_] = N if 0.0 <= score <= 0.4 \
        else NN if 0.4 < score <= 0.6 \
        else P


stop_words = open('stop_words.txt', 'r').read().split(',')

total_training_instances = 0
training_instances = defaultdict(int)
prior = dict()
likelihood = defaultdict(lambda: defaultdict(int))
class_word_count = defaultdict(int)

for id_ in parsed_data.keys()[:number_of_train_data]:
    current_phrase = parsed_data[id_]
    current_label = labels_dict[parsed_label[id_]]

    current_words = filter(lambda word_: word_ not in stop_words, re.findall(r'[a-zA-Z][a-zA-Z]+', current_phrase))

    if len(current_words) == 0:
        continue

    total_training_instances += 1
    training_instances[current_label] += 1

    for word in current_words:
        likelihood[current_label][word.lower()] += 1
        class_word_count[current_label] += 1


for label in labels_dict.values():
    prior[label] = training_instances[label] / total_training_instances

    for word in likelihood[label].keys():
        likelihood[label][word] /= class_word_count[label]


additive_smoothing = 10 ** -100


def find_likelihood(id_, word_):
    try:
        return likelihood[id_][word_] + additive_smoothing
    except KeyError:
        return 0 + additive_smoothing


def posterior(phrase):
    results = defaultdict(float)

    for id in labels_dict.values():
        log_likelihood = log(prior[id])

        for word_ in phrase:
            log_likelihood += log(find_likelihood(id, word_.lower()))

        results[id] = log_likelihood

    results = sorted(results.items(), key=operator.itemgetter(1), reverse=True)

    return results[0]


correctly_predicted = 0
total_test_data = 0

# print posterior('works and qualifies as cool at times'.split(' '))

for id_ in parsed_data.keys()[number_of_train_data:]:
    current_phrase = parsed_data[id_]
    current_label = labels_dict[parsed_label[id_]]

    current_words = filter(lambda word_: word_ not in stop_words, re.findall(r'[a-zA-Z][a-zA-Z]+', current_phrase))

    if len(current_words) == 0:
        continue

    predicted_label = posterior(current_words)

    if predicted_label[0] == current_label:
        correctly_predicted += 1

    total_test_data += 1


print (correctly_predicted/total_test_data) * 100

