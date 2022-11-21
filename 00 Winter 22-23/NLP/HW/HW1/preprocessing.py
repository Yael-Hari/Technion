from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

WORD = 0
TAG = 1


class FeatureStatistics:
    F200 = True
    F300 = True

    def __init__(self, f200=True, f300=True):
        self.n_total_features = 0  # Total number of features accumulated
        FeatureStatistics.F200 = f200
        FeatureStatistics.F300 = f300
        # Init all features dictionaries
        # the feature classes used in the code
        feature_dict_list = [f"f10{i}" for i in range(8)]
        if FeatureStatistics.F200:
            feature_dict_list += [f"f20{i}" for i in range(6)]
        if FeatureStatistics.F300:
            feature_dict_list += [f"f30{i}" for i in range(10)] + [
                f"f3{i}" for i in range(10, 13)
            ]
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        """
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        """
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(
            int
        )  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(
            int
        )  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the text
        self.n_tags = 0
        self.tags_list = []

    def increment_val_in_feature_dict(self, feature: str, val: tuple):
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
                split_words = line.split(" ")

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                num_of_words = len(sentence)
                for word_idx in range(2, num_of_words - 1):
                    cur_word, cur_tag = sentence[word_idx]
                    p_word, p_tag = sentence[word_idx - 1]
                    pp_word, pp_tag = sentence[word_idx - 2]
                    n_word, _ = sentence[word_idx + 1]
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1
                    cur_word_len = len(cur_word)
                    p_word_len = len(p_word)
                    n_word_len = len(n_word)

                    if cur_tag not in self.tags_list:
                        self.tags_list.append(cur_tag)
                        self.n_tags += 1

                    # ~~~~~~~~~~ COUNT FOR EVERY FEATURE FAMILIES ~~~~~~~~~~
                    # f100 - cont appearances of <word, tags> tuples
                    word_tag = (cur_word.lower(), cur_tag)
                    self.increment_val_in_feature_dict("f100", word_tag)
                    # f101 - suffix <=4 and tag pairs
                    for i in range(1, cur_word_len):
                        suffix_tag = (cur_word[-i:].lower(), cur_tag)
                        self.increment_val_in_feature_dict("f101", suffix_tag)
                    # f102 - prefix <=4 and tag pairs
                    for i in range(1, cur_word_len):
                        prefix_tag = (cur_word[:i].lower(), cur_tag)
                        self.increment_val_in_feature_dict("f102", prefix_tag)
                    # f103 - trigram tags
                    trigram_tags = (pp_tag, p_tag, cur_tag)
                    self.increment_val_in_feature_dict("f103", trigram_tags)
                    # f104
                    bigram_tags = (p_tag, cur_tag)
                    self.increment_val_in_feature_dict("f104", bigram_tags)
                    # f105 - count appearances of <tag>
                    self.increment_val_in_feature_dict("f105", cur_tag)
                    # f106 - previous word and tag pairs
                    prev_word_tag = (p_word.lower(), cur_tag)
                    self.increment_val_in_feature_dict("f106", prev_word_tag)
                    # f107 - next word and tag pairs
                    next_word_tag = (n_word.lower(), cur_tag)
                    self.increment_val_in_feature_dict("f107", next_word_tag)

                    if FeatureStatistics.F200:
                        # ~~~~~~~~~~ COUNT FOR FEATURES FOR CAPITAL LETTERS AND DIGITS HANDLING ~~~~

                        # f200 - (bool: is starting with capital letter, tag)
                        first_letter_is_capital = cur_word[0].isupper()
                        is_first_capital_true_tag = (first_letter_is_capital, cur_tag)
                        self.increment_val_in_feature_dict(
                            "f200", is_first_capital_true_tag
                        )

                        # f201 - (bool: is first word in sentence, tag)
                        is_first_word = (p_word == "*") and (pp_word == "*")
                        f201_tuple = (is_first_word, cur_tag)
                        self.increment_val_in_feature_dict("f201", f201_tuple)

                        (
                            alphabetical_cnt,
                            capital_letter_cnt,
                            digits_cnt,
                        ) = get_alpha_capital_digits_counts(cur_word)

                        # f202 - (bool: has more than 1 capital letter, tag)
                        has_more_than_one_capital = capital_letter_cnt > 1
                        f202_tuple = (has_more_than_one_capital, cur_tag)
                        self.increment_val_in_feature_dict("f202", f202_tuple)

                        # f203 - (bool: has exactly 1 capital letter no matter where, tag)
                        has_exactly_one_capital = capital_letter_cnt == 1
                        f203_tuple = (has_exactly_one_capital, cur_tag)
                        self.increment_val_in_feature_dict("f203", f203_tuple)

                        # f204 - (bool: is a number, tag)
                        is_number = cur_word.isnumeric()
                        f204_tuple = (is_number, cur_word)
                        self.increment_val_in_feature_dict("f204", f204_tuple)

                        # f205 - (bool: has a digit and a letter, tag)
                        has_letter_and_digit = (alphabetical_cnt > 0) and (
                            digits_cnt > 0
                        )
                        f205_tuple = (has_letter_and_digit, cur_tag)
                        self.increment_val_in_feature_dict("f205", f205_tuple)

                    if FeatureStatistics.F300:
                        # ~~~~~~~~~~ OUR ADDED SPECIAL FEATURES  ~~~~~~~~~~

                        # f300 - (suffix of prev word, tag_curr)
                        for i in range(1, p_word_len):
                            p_suffix_tag = (p_word[-i:], cur_tag)
                            self.increment_val_in_feature_dict("f300", p_suffix_tag)

                        # f301 - (prefix of prev word, tag_curr)
                        for i in range(1, p_word_len):
                            p_prefix_tag = (p_word[:i], cur_tag)
                            self.increment_val_in_feature_dict("f301", p_prefix_tag)

                        # f302 - (suffix of next word, tag_curr)
                        for i in range(1, n_word_len):
                            n_suffix_tag = (n_word[-i:], cur_tag)
                            self.increment_val_in_feature_dict("f302", n_suffix_tag)

                        # f303 - (prefix of next word, tag_curr)
                        for i in range(1, p_word_len):
                            p_prefix_tag = (p_word[:i], cur_tag)
                            self.increment_val_in_feature_dict("f303", p_prefix_tag)

                        # f304 - (curr word contains a punctuation mark, tag_curr)
                        puncs = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
                        c_word_has_puncs = (cur_word.strip(puncs) != cur_word) and (
                            len(cur_word.strip(puncs)) > 0
                        )
                        f304_tuple = (c_word_has_puncs, cur_tag)
                        self.increment_val_in_feature_dict("f304", f304_tuple)

                        # f305 - (curr word contains a dots, tag_curr)
                        c_word_has_dot = ("." in cur_word) and (len(cur_word) > 0)
                        f305_tuple = (c_word_has_dot, cur_tag)
                        self.increment_val_in_feature_dict("f305", f305_tuple)

                        # f306 - (bigram of pp_tag and cur tag, tag curr)
                        pp_bigram_tags = (pp_tag, cur_tag)
                        self.increment_val_in_feature_dict("f306", pp_bigram_tags)

                        # f307 - (is word contains "x", tag curr)
                        c_word_has_x = "x" in cur_word
                        f307_tuple = (c_word_has_x, cur_tag)
                        self.increment_val_in_feature_dict("f307", f307_tuple)

                        # f308 - (is word starts with "z", tag curr)
                        c_word_starts_with_z = cur_word[0] == "z"
                        f308_tuple = (c_word_starts_with_z, cur_tag)
                        self.increment_val_in_feature_dict("f308", f308_tuple)

                        # f309 - (is 2nd word in sentence, tag curr)
                        is_2nd_word = (p_word != "*") and (pp_word == "*")
                        f309_tuple = (is_2nd_word, cur_tag)
                        self.increment_val_in_feature_dict("f309", f309_tuple)

                        # f310 - (is last word in sentence, tag curr)
                        is_last_word = n_word == "~"
                        f310_tuple = (is_last_word, cur_tag)
                        self.increment_val_in_feature_dict("f310", f310_tuple)

                        # f311 - (is not 1st\2nd\last word in sentence, tag curr)
                        is_middle_word = (
                            (p_word != "*") and (pp_word != "*") and (n_word != "~")
                        )
                        f311_tuple = (is_middle_word, cur_tag)
                        self.increment_val_in_feature_dict("f311", f311_tuple)

                        # f312 - (first_letter_is_capital, suffix)
                        for i in range(1, cur_word_len):
                            f312_tuple = (first_letter_is_capital, cur_word[-i:])
                            self.increment_val_in_feature_dict("f312", f312_tuple)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # create history
                    history = (
                        cur_word,
                        cur_tag,
                        p_word,
                        p_tag,
                        pp_word,
                        pp_tag,
                        n_word,
                    )
                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        # statistics class, for each feature gives empirical counts
        self.feature_statistics = feature_statistics
        # feature count threshold - empirical count must be higher than this
        self.threshold = threshold

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {f"f10{i}": OrderedDict() for i in range(8)}
        if FeatureStatistics.F200:
            self.feature_to_idx = self.merge_dicts(
                [
                    self.feature_to_idx,
                    {f"f20{i}": OrderedDict() for i in range(6)},
                ]
            )
        if FeatureStatistics.F300:
            self.feature_to_idx = self.merge_dicts(
                [
                    self.feature_to_idx,
                    {f"f30{i}": OrderedDict() for i in range(10)},
                    {f"f3{i}": OrderedDict() for i in range(10, 13)},
                ]
            )

        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = (
            OrderedDict()
        )  # Dict[(tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}): [relevant_features_indexes]]
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix
        self.tags_list = feature_statistics.tags_list
        self.n_tags = feature_statistics.n_tags

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
            for feat, count in self.feature_statistics.feature_rep_dict[
                feat_class
            ].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process -
        self.big_matrix and self.small_matrix
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
                demi_hist = (
                    hist[0],
                    y_tag,
                    hist[2],
                    hist[3],
                    hist[4],
                    hist[5],
                    hist[6],
                )
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix(
            (np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
            shape=(
                len(self.feature_statistics.tags)
                * len(self.feature_statistics.histories),
                self.n_total_features,
            ),
            dtype=bool,
        )
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool,
        )


def represent_input_with_features(
    history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple, int]]
) -> List[int]:
    """
    Extract feature vector in per a given history
    @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
    @param dict_of_dicts: a dictionary of each feature and the index it was given
    @return a list with all features that are relevant to the given history
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    cur_word_len = len(c_word)
    p_word_len = len(p_word)
    n_word_len = len(n_word)
    features = []

    # ~~~~~~~~~~ LOCAL FUNC FOR UPDATING FEATURES ~~~~~~~~~~

    def update_features(feature_name: str, feature_tuple: tuple) -> None:
        if feature_tuple in dict_of_dicts[feature_name]:
            features.append(dict_of_dicts[feature_name][feature_tuple])

    # ~~~~~~~~~~ RATNAPARKHI FEATURES ~~~~~~~~~~

    # f100 - word / tag pairs
    update_features("f100", (c_word.lower(), c_tag))

    # f101 - suffix <=4 and tag pairs
    for i in range(1, cur_word_len):
        c_suffix_tag = (c_word[-i:], c_tag)
        update_features("f101", c_suffix_tag)

    # f102 - prefix <=4 and tag pairs
    for i in range(1, cur_word_len):
        c_prefix_tag = (c_word[:i], c_tag)
        update_features("f102", c_prefix_tag)

    # f103 - trigram tags
    trigram_tags = (pp_tag, p_tag, c_tag)
    update_features("f103", trigram_tags)

    # f104 - bigram tags
    bigram_tags = (p_tag, c_tag)
    update_features("f104", bigram_tags)

    # f105 - unigram tag
    update_features("f105", c_tag)

    # f106 - previous word and tag pairs
    prev_word_tag = (p_word, c_tag)
    update_features("f106", prev_word_tag)

    # f107 - next word and tag pairs
    next_word_tag = (n_word, c_tag)
    update_features("f107", next_word_tag)

    if "f200" in dict_of_dicts:
        # ~~~~~~~~~~ ADDED FEATURES FOR CAPITAL LETTERS AND DIGITS HANDLING ~~~~~~~~~~

        # f200 - (bool: is starting with capital letter, tag)
        first_letter_is_capital = c_word[0].isupper()
        f200_tuple = (first_letter_is_capital, c_tag)
        update_features("f200", f200_tuple)

        (
            alphabetical_cnt,
            capital_letter_cnt,
            digits_cnt,
        ) = get_alpha_capital_digits_counts(c_word)

        # f201 - (is first word in sentence, tag)
        is_first_word = (p_word == "*") and (pp_word == "*")
        f201_tuple = (is_first_word, c_tag)
        update_features("f201", f201_tuple)

        # f202 - (bool: has more than 1 capital letter, tag)
        has_more_than_one_capital = capital_letter_cnt > 1
        f202_tuple = (has_more_than_one_capital, c_tag)
        update_features("f202", f202_tuple)

        # f203 - (bool: has exactly 1 capital letter no matter where, tag)
        has_exactly_one_capital = capital_letter_cnt == 1
        f203_tuple = (has_exactly_one_capital, c_tag)
        update_features("f203", f203_tuple)

        # f204 - (bool: is a number, tag)
        is_number = c_word.isnumeric()
        f204_tuple = (is_number, c_tag)
        update_features("f204", f204_tuple)

        # f205 - (bool: has a digit and a letter, tag)
        has_letter_and_digit = (alphabetical_cnt > 0) and (digits_cnt > 0)
        f205_tuple = (has_letter_and_digit, c_tag)
        update_features("f205", f205_tuple)

    if "f300" in dict_of_dicts:
        # ~~~~~~~~~~ OUR ADDED SPECIAL FEATURES  ~~~~~~~~~~

        # f300 - (suffix of prev word, tag_curr)
        for i in range(1, p_word_len):
            p_suffix_tag = (p_word[-i:], c_tag)
            update_features("f300", p_suffix_tag)

        # f301 - (prefix of prev word, tag_curr)
        for i in range(1, p_word_len):
            p_prefix_tag = (p_word[:i], c_tag)
            update_features("f301", p_prefix_tag)

        # f302 - (suffix of next word, tag_curr)
        for i in range(1, n_word_len):
            n_suffix_tag = (n_word[-i:], c_tag)
            update_features("f302", n_suffix_tag)

        # f303 - (prefix of next word, tag_curr)
        for i in range(1, p_word_len):
            p_prefix_tag = (p_word[:i], c_tag)
            update_features("f303", p_prefix_tag)

        # f304 - (curr word contains a punctuation mark, tag_curr)
        puncs = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        c_word_has_puncs = (c_word.strip(puncs) != c_word) and (
            len(c_word.strip(puncs)) > 0
        )
        f304_tuple = (c_word_has_puncs, c_tag)
        update_features("f304", f304_tuple)

        # f305 - (curr word contains a dots, tag_curr)
        c_word_has_dot = ("." in c_word) and (len(c_word) > 0)
        f305_tuple = (c_word_has_dot, c_tag)
        update_features("f305", f305_tuple)

        # f306 - (bigram of pp_tag and cur tag, tag curr)
        pp_bigram_tags = (pp_tag, c_tag)
        update_features("f306", pp_bigram_tags)

        # f307 - (is word contains "x", tag curr)
        c_word_has_x = "x" in c_word
        f307_tuple = (c_word_has_x, c_tag)
        update_features("f307", f307_tuple)

        # f308 - (is word starts with "z", tag curr)
        c_word_starts_with_z = c_word[0] == "z"
        f308_tuple = (c_word_starts_with_z, c_tag)
        update_features("f308", f308_tuple)

        # f309 - (is 2nd word in sentence, tag curr)
        is_2nd_word = (p_word != "*") and (pp_word == "*")
        f309_tuple = (is_2nd_word, c_tag)
        update_features("f309", f309_tuple)

        # f310 - (is last word in sentence, tag curr)
        is_last_word = n_word == "~"
        f310_tuple = (is_last_word, c_tag)
        update_features("f310", f310_tuple)

        # f311 - (is not 1st\2nd\last word in sentence, tag curr)
        is_middle_word = (p_word != "*") and (pp_word != "*") and (n_word != "~")
        f311_tuple = (is_middle_word, c_tag)
        update_features("f311", f311_tuple)

        # f312 - (first_letter_is_capital, suffix)
        for i in range(1, cur_word_len):
            f312_tuple = (first_letter_is_capital, c_word[-i:])
            update_features("f312", f312_tuple)

    return features


def preprocess_train(train_path, threshold, f200, f300):
    # Statistics
    statistics = FeatureStatistics(f200, f300)
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
    @return: a list of all the sentences, each sentence represented
            as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(" ")
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split("_")
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
