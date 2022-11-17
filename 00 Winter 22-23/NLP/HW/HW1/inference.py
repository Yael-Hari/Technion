from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from preprocessing import read_test


def get_top_B_idx(Matrix: np.array, B: int) -> List[np.array]:
    # TODO: complete
    """return B_best_idx"""
    m = Matrix.copy()
    B_best_idx = []

    for i in range(B):
        cur_max = np.unravel_index(m.argmax(), m.shape)
        B_best_idx.append(cur_max)
        m[cur_max] = 0

    return B_best_idx


def check_if_known_word(word):
    # check number
    if word.isdigit():
        return 'CD'

    # check known tags
    known_tags_dict = {
        ',': ',', "``": "``", 'The': 'DT', "$": "$",
        "''": "''", "in": "IN", "a": "DT", "A": "DT",
        ":": ":", ";": ":", "--": ":", "of": "IN",
        "to": "TO", "from": "IN", "for": "IN", 'because': 'IN',
        "than": 'IN', "that": 'IN', "at": 'IN', "as": 'IN',
        "into": 'IN', "by": 'IN', "on": 'IN',
    }
    if word in known_tags_dict.keys():
        return known_tags_dict[word]

    return None


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    """
    @ n_words: number of words in the sentence
    @ v_tag: tag of current word           (position k)
    @ u_tag: tag of previous word          (position k-1)
    @ t_tag: tag of previous-previous word (position k-2)
    @ x: sentence
    @ sentence[i] == x_i : word i in the sentence
    @ weights: pre_trained_weights

    @ Pi: matrix of size n_words, n_tags, n_tags
        Pi[k][u][v] = maximum probability of a tag sequence ending in tags u, v at position k
    @ Bp: matrix of size n_words, n_tags, n_tags
        BP[k][u][v] = argmax of probability of a tag sequence ending in tags u, v at position k
    @ Qv: np array
        Qv[v] = exp(w * feature_vec(v |k_history))
        Qv[u][v] = exp(w * feature_vec(v |u, k_history))
        Qv[t][u][v] = exp(w * feature_vec(v |t, u, k_history))
        -----
        Q(history) = probabilty that the current word will have the tag v, given
               the k_history: (x_k, v_tag, x_k-1, u_tag, x_k-2, t_tag, x_k+1)
        Q(history) = exp(w * feature_vec(v |k_history)
             / sum_over_all_tags_v'[exp(-- same as above --)]
        ==   exp(sum the cordinates of w in the relevant indexes of feature_vec)
             / sum_over_all_tags_v'[exp(-- same as above --)]
    """
    tags_list = feature2id.tags_list  # list of all possible tags
    n_tags = feature2id.n_tags + 1  # number of tags in train set, +1 for "*"
    x = sentence[2:-1]  # last letter is always '~'
    weights = np.array(pre_trained_weights)
    n_words = len(x)
    Pi = np.zeros([n_words, n_tags, n_tags])
    Bp = np.zeros([n_words, n_tags, n_tags])
    pred_tags = np.array(len(sentence))
    B = 3  # Beam search parameter
    B_best_idx = []
    dict_tag_to_idx = {v_tag: v_idx for v_idx, v_tag in enumerate(tags_list)}
    star_idx = n_tags - 1   # "*"

    # histories_features: OrderedDict[history_tuple: [relevant_features_indexes]]
    histories_features = feature2id.histories_features

    for k in range(n_words):
        known_tag = check_if_known_word(x[k])
        # ------------------ Calc Pi k = 0 ----------------------
        if k == 0:
            # by definition:
            # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
            # Pi(0, "*", v) = Pi(-1, "*", "*") * Q(k_history)
            # --> Pi(0, "*", v) = 1* Q(k_history)

            # check if known word:
            if known_tag:
                v_idx = dict_tag_to_idx[known_tag]
                Pi[k][star_idx][v_idx] = 1
                Bp[k][star_idx][v_idx] = star_idx
            else:
                # calculate Q and we'll use it later a lot
                Qv = np.zeros(n_tags)
                Q_all_v_tags = 0
                for v_idx, v_tag in enumerate(tags_list):
                    # history tuple: (x_k, v_tag, x_k-1, u_tag, x_k-2, t_tag, x_k+1)
                    history_tuple = (x[k], v_tag, "*", "*", "*", "*", x[k + 1])
                    relevant_idx = histories_features[history_tuple]
                    Qv[v_idx] = np.exp(weights[relevant_idx].sum())
                    Q_all_v_tags += Qv[v_idx]

                # calculate Pi
                for v_idx, v_tag in enumerate(tags_list):
                    Pi[k][star_idx][v_idx] = Qv[v_idx] / Q_all_v_tags
                    Bp[k][star_idx][v_idx] = star_idx

        # ------------------ Calc Pi k = 1 ----------------------
        if k == 1:
            # by definition:
            # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
            # Pi(1, u, v) = Pi(0, "*", u) * Q(k_history)

            # check if known word and if so, put only the known tag in the possible tags for v:
            if known_tag:
                v_idx = dict_tag_to_idx[known_tag]
                v_tags_list = [v_idx]
            else:
                v_tags_list = tags_list

            # calculate Q and we'll use it later a lot
            Qv = np.zeros([n_tags, n_tags])
            Q_all_v_tags = np.zeros(n_tags)
            for u_idx, u_tag in enumerate(tags_list):
                for v_idx, v_tag in enumerate(v_tags_list):
                    # history tuple: (x_k, v, x_k-1, u, x_k-2, t, x_k+1)
                    history_tuple = (x[k], v_tag, x[k - 1], u_tag, "*", "*", x[k + 1])
                    relevant_idx = histories_features[history_tuple]
                    Qv[u_idx][v_idx] = np.exp(weights[relevant_idx].sum())
                    Q_all_v_tags[u_idx] += Qv[u_idx][v_idx]

            # calculate Pi
            for u_idx, u_tag in enumerate(tags_list):
                for v_idx, v_tag in enumerate(v_tags_list):
                    Pi[k][u_idx][v_idx] = (
                        Pi[k - 1][star_idx][u_idx] * Qv[u_idx][v_idx] / Q_all_v_tags[u_idx]
                    )
                    Bp[k][u_idx][v_idx] = star_idx

            # find Top B Pi values indexes
            B_best_idx = get_top_B_idx(Pi[k], B)

        # ------------------ Calc Pi k >= 2 ----------------------
        else:
            # by definition:
            # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)

            # check if known word and if so, put only the known tag in the possible tags for v:
            if known_tag:
                v_idx = dict_tag_to_idx[known_tag]
                v_tags_list = [v_idx]
            else:
                v_tags_list = tags_list

            # calculate Q and we'll use it later a lot
            Qv = np.zeros([n_tags, n_tags, n_tags])
            Q_all_v_tags = np.zeros([n_tags, n_tags])
            for t_idx, u_idx in B_best_idx:
                t_tag = tags_list[t_idx]
                u_tag = tags_list[u_idx]
                for v_idx, v_tag in enumerate(v_tags_list):
                    # history tuple: (x_k, v, x_k-1, u, x_k-2, t, x_k+1)
                    history_tuple = (
                        x[k],
                        v_tag,
                        x[k - 1],
                        u_tag,
                        x[k - 2],
                        t_tag,
                        x[k + 1],
                    )
                    relevant_idx = histories_features[history_tuple]
                    Qv[t_idx][u_idx][v_idx] = np.exp(weights[relevant_idx].sum())
                    Q_all_v_tags[t_idx][u_idx] += Qv[t_idx][u_idx][v_idx]

            # calculate Pi
            for t_idx, u_idx in B_best_idx:
                for v_idx, v_tag in enumerate(v_tags_list):
                    t_scores = np.zeros(n_tags)
                    t_scores[t_idx] = (
                        Pi[k - 1][t_idx][u_idx]
                        * Qv[t_idx][u_idx][v_idx]
                        / Q_all_v_tags[t_idx][u_idx]
                    )
                    Pi[k][u_idx][v_idx] = np.max(t_scores)
                    argmax_tag = tags_list[np.argmax(t_scores)]
                    Bp[k][u_idx][v_idx] = dict_tag_to_idx[argmax_tag]

            # find Top B Pi values indexes
            B_best_idx = get_top_B_idx(Pi[k], B)

    # ------------------ Get Predicted Tags by Bp ----------------------
    # last 2 words in sentence
    pred_tags[n_words - 1] = tags_list[np.argmax(Pi[n_words - 1])]
    pred_tags[n_words - 2] = tags_list[np.argmax(Pi[n_words - 2])]
    for k in reversed(range(n_words - 2)):
        pred_tag_idx = Bp[k + 2][pred_tags[k + 1]][pred_tags[k + 2]]
        pred_tags[k] = dict_tag_to_idx[pred_tag_idx]

    pred_tags[n_words] = "~"
    return pred_tags


def find_n_argmin_idx(values_list: list, n: int):
    values_list = values_list.copy()
    n_argmin_idx = []
    n = min(n, len(values_list))
    for i in range(n):
        n_argmin_idx.append(np.argmin(values_list))
        values_list.remove(np.min(values_list))
    return n_argmin_idx


def print_10_tags_with_lowest_val(score_method: dict):
    vals_list = list(score_method.values())
    keys_list = list(score_method.keys())
    argmin_10_idx = find_n_argmin_idx(vals_list, n=10)
    for argmin_idx in argmin_10_idx:
        tag = keys_list[argmin_idx]
        val = vals_list[argmin_idx]
        print(f"10 tags with the lowest {score_method}: \n")
        print(f"{tag=}, {val=} \n")


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    # prepare for test evaluation
    if tagged:
        true_list = []
        pred_list = []
        tags_list = []
        Tp = {}
        Fp = {}
        Fn = {}
        precision = {}
        recall = {}
        f1 = {}
        n_preds = 0

    # go over each sentence in test set
    for k, sen in tqdm(enumerate(test), total=len(test)):
        # take only words without tags
        sentence = sen[0]
        # get pred
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        # remove * *
        sentence = sentence[2:]
        # write preds to file
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")

        # add details tp fp ...
        if tagged:
            true_tags = sen[1]
            pred_tags = pred
            n_words = len(true_tags)
            for k in range(n_words):
                true = true_tags[k]
                pred = pred_tags[k]
                n_preds += 1
                true_list.append(true)
                pred_list.append(pred)

                # if new tag add relevant details
                if true not in tags_list:
                    tags_list.append(true)
                    Tp[true] = 0
                    Fp[true] = 0
                    Fn[true] = 0
                if pred not in tags_list:
                    tags_list.append(pred)
                    Tp[pred] = 0
                    Fp[pred] = 0
                    Fn[pred] = 0

                if true == pred:
                    Tp[true] += 1
                else:  # true != pred:
                    Fn[true] += 1  # didn't catch the true tag
                    Fp[pred] += 1  # pred is the wrong tag

    output_file.close()

    # ------------------------- calc evaluations -------------------------
    if tagged:
        # accuracy
        accuracy = sum(list(Tp.values())) / n_preds

        # precision, recall, f1 for each tag
        for tag in tags_list:
            precision[tag] = Tp[tag] / (Tp[tag] + Fp[tag])
            recall[tag] = Tp[tag] / (Tp[tag] + Fn[tag])
            f1[tag] = (
                2 * (precision[tag] * recall[tag]) / (precision[tag] + recall[tag])
            )

        # f1
        mean_f1 = np.mean(list(f1.values()))
        median_f1 = np.median(list(f1.values()))
        print(f"{accuracy=}, {mean_f1=}, {median_f1=}")

        # 10 tags with the lowest precision
        print_10_tags_with_lowest_val(precision)
        # 10 tags with the lowest recall
        print_10_tags_with_lowest_val(recall)
        # 10 tags with the lowest f1 score
        print_10_tags_with_lowest_val(f1)

        # confusion matrix
        # disp = ConfusionMatrixDisplay.from_predictions(true_list, pred_list, normalize="false")
        # plt.show()
        # disp = ConfusionMatrixDisplay.from_predictions(true_list, pred_list, normalize="true")
        # plt.show()
