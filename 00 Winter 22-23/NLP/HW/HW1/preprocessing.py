from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = [f'f10{i}' for i in range(8)] + \
            [f'f20{i}' for i in range(6)]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the text

    def increment_val_in_feature_dict(self, val: tuple, feature: str):
        if val not in self.feature_rep_dict[feature]:
            self.feature_rep_dict[feature][val] = 1
        else:
            self.feature_rep_dict[feature][val] += 1

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                num_of_words = len(sentence)
                for word_idx in range(2, num_of_words-1):
                    cur_word, cur_tag = sentence[word_idx]
                    p_word, p_tag = sentence[word_idx-1]
                    pp_word, pp_tag = sentence[word_idx-2]
                    n_word, _ = sentence[word_idx+1]
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1
                    cur_word_len = len(cur_word)

                    # ~~~~~~~~~~ COUNT FOR EVERY FEATURE FAMILIES ~~~~~~~~~~
                    # f100 - cont appearances of <word, tags> tuples
                    word_tag = (cur_word.lower(), cur_tag)
                    self.increment_val_in_feature_dict(word_tag, "f100")
                    # f101 - suffix <=4 and tag pairs
                    for i in range(1, cur_word_len):
                        suffix_tag = (cur_word[-i:].lower(), cur_tag)
                        self.increment_val_in_feature_dict(suffix_tag, "f101")
                    # f102 - prefix <=4 and tag pairs
                    for i in range(1, cur_word_len):
                        prefix_tag = (cur_word[:i].lower(), cur_tag)
                        self.increment_val_in_feature_dict(prefix_tag, "f102")
                    # f103 - trigram tags
                    trigram_tags = (pp_tag, p_tag, cur_tag)
                    self.increment_val_in_feature_dict(trigram_tags, "f103")
                    # f104
                    bigram_tags = (p_tag, cur_tag)
                    self.increment_val_in_feature_dict(bigram_tags, 'f104')
                    # f105 - count appearances of <tag>
                    self.increment_val_in_feature_dict(cur_tag, 'f105')
                    # f106 - previous word and tag pairs
                    prev_word_tag = (p_word.lower(), cur_tag)
                    self.increment_val_in_feature_dict(prev_word_tag, "f106")
                    # f107 - next word and tag pairs
                    next_word_tag = (n_word.lower(), cur_tag)
                    self.increment_val_in_feature_dict(next_word_tag, "f107")

                    # ~~~~~~~~~~ COUNT FOR FEATURES FOR CAPITAL LETTERS AND DIGITS HANDLING ~~~~~~~~~~
                    # f200 - (bool: is starting with capital letter, tag)
                    first_letter_is_capital = cur_word[0].isupper()
                    is_first_capital_true_tag = (first_letter_is_capital, cur_tag)
                    self.increment_val_in_feature_dict(is_first_capital_true_tag, "f200")

                    # f201 - (bool: is first word in sentence, tag)
                    is_first_word = (p_word == "*") and (pp_word == "*")
                    f201_tuple = (is_first_word, cur_tag)
                    self.increment_val_in_feature_dict(f201_tuple, "f201")

                    alphabetical_cnt, capital_letter_cnt, digits_cnt = get_alpha_capital_digits_counts(cur_word)

                    # f202 - (bool: has more than 1 capital letter, tag)
                    has_more_than_one_capital = (capital_letter_cnt > 1)
                    f202_tuple = (has_more_than_one_capital, cur_tag)
                    self.increment_val_in_feature_dict(f202_tuple, "f202")

                    # f203 - (bool: has exactly 1 capital letter no matter where, tag)
                    has_exactly_one_capital = (capital_letter_cnt == 1)
                    f203_tuple = (has_exactly_one_capital, cur_tag)
                    self.increment_val_in_feature_dict(f203_tuple, "f203")

                    # f204 - (bool: is a number, tag)
                    is_number = cur_word.isnumeric()
                    f204_tuple = (is_number, cur_word)
                    self.increment_val_in_feature_dict(f204_tuple, "f204")

                    # f205 - (bool: has a digit and a letter, tag)
                    has_letter_and_digit = (alphabetical_cnt > 0) and (digits_cnt > 0)
                    f205_tuple = (has_letter_and_digit, cur_tag)
                    self.increment_val_in_feature_dict(f205_tuple, "f205")

                    # create history
                    history = (cur_word, cur_tag, p_word, p_tag, pp_word, pp_tag, n_word)
                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = self.merge_dicts(
            [
                {f"f10{i}": OrderedDict() for i in range(8)},
                {f"f20{i}": OrderedDict() for i in range(6)},
            ]
        )
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()  # Dict[(tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}): [relevant_features_indexes]]
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def merge_dicts(self, dict_list: List[Dict]) -> dict:
        result_dict = dict_list[0]
        for d in dict_list[1:]:
            result_dict.update(d)
        return result_dict

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple, int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    cur_word_len = len(c_word)
    features = []

    # ~~~~~~~~~~ RATNAPARKHI FEATURES ~~~~~~~~~~

    # f100 - word / tag pairs
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # f101 - suffix <=4 and tag pairs
    for i in range(1, cur_word_len):
        suffix_tag = (c_word[-i:], c_tag)
        if suffix_tag in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][suffix_tag])

    # f102 - prefix <=4 and tag pairs
    for i in range(1, cur_word_len):
        prefix_tag = (c_word[:i], c_tag)
        if prefix_tag in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][prefix_tag])

    # f103 - trigram tags
    trigram_tags = (pp_tag, p_tag, c_tag)
    if trigram_tags in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][trigram_tags])

    # f104 - bigram tags
    bigram_tags = (p_tag, c_tag)
    if bigram_tags in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][bigram_tags])

    # f105 - unigram tag
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])

    # f106 - previous word and tag pairs
    prev_word_tag = (p_word, c_tag)
    if prev_word_tag in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][prev_word_tag])

    # f107 - next word and tag pairs
    next_word_tag = (n_word, c_tag)
    if next_word_tag in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][next_word_tag])

    # ~~~~~~~~~~ ADDED FEATURES FOR CAPITAL LETTERS AND DIGITS HANDLING ~~~~~~~~~~

    # f200 - (bool: is starting with capital letter, tag)
    first_letter_is_capital = c_word[0].isupper()
    f200_tuple = (first_letter_is_capital, c_tag)
    if f200_tuple in dict_of_dicts["f200"]:
        features.append(dict_of_dicts["f200"][f200_tuple])

    alphabetical_cnt, capital_letter_cnt, digits_cnt = get_alpha_capital_digits_counts(c_word)

    # f201 - (is first word in sentence, tag)
    is_first_word = (p_word == "*") and (pp_word == "*")
    f201_tuple = (is_first_word, c_tag)
    if f201_tuple in dict_of_dicts["f201"]:
        features.append(dict_of_dicts["f201"][f201_tuple])

    # f202 - (bool: has more than 1 capital letter, tag)
    has_more_than_one_capital = capital_letter_cnt > 1
    f202_tuple = (has_more_than_one_capital, c_tag)
    if f202_tuple in dict_of_dicts["f202"]:
        features.append(dict_of_dicts["f202"][f202_tuple])

    # f203 - (bool: has exactly 1 capital letter no matter where, tag)
    has_exactly_one_capital = (capital_letter_cnt == 1)
    f203_tuple = (has_exactly_one_capital, c_tag)
    if f203_tuple in dict_of_dicts["f203"]:
        features.append(dict_of_dicts["f203"][f203_tuple])

    # f204 - (bool: is a number, tag)
    is_number = c_word.isnumeric()
    f204_tuple = (is_number, c_tag)
    if f204_tuple in dict_of_dicts["f204"]:
        features.append(dict_of_dicts["f204"][f204_tuple])

    # f205 - (bool: has a digit and a letter, tag)
    has_letter_and_digit = (alphabetical_cnt > 0) and (digits_cnt > 0)
    f205_tuple = (has_letter_and_digit, c_tag)
    if f205_tuple in dict_of_dicts["f205"]:
        features.append(dict_of_dicts["f205"][f205_tuple])

    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences


def get_alpha_capital_digits_counts(c_word: str) -> Tuple[int, int, int]:
    # TODO: check if there is a efficient way to check instead of the following:
        alphabetical_cnt = 0
        capital_letter_cnt = 0
        digits_cnt = 0
        for letter in c_word:
            if letter.isalpha():
                alphabetical_cnt += 1
                if letter.isupper():
                    capital_letter_cnt += 1
            if letter.isdigit():
                digits_cnt += 1

        return alphabetical_cnt, capital_letter_cnt, digits_cnt

