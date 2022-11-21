from typing import List

import numpy as np
# import pandas as pd
from tqdm import tqdm

from preprocessing import read_test, represent_input_with_features


def get_top_B_idx_dict(Matrix: np.array, B: int) -> List[np.array]:
    """return B_best_idx"""
    m = Matrix.copy()
    B_best_idx = []

    for i in range(B):
        cur_max = np.unravel_index(m.argmax(), m.shape)
        B_best_idx.append(cur_max)
        m[cur_max] = 0

    u_to_t_best_dict = dict()
    for t, u in B_best_idx:
        if u not in u_to_t_best_dict.keys():
            u_to_t_best_dict[u] = [t]
        else:
            u_to_t_best_dict[u].append(t)
    return u_to_t_best_dict


def check_if_known_word(word):
    # check number
    if word.isdigit():
        return "CD"

    # check known tags
    known_tags_dict = {
        ".": ".",
        "?": ".",
        ",": ",",
        "``": "``",
        "$": "$",
        "#": "#",
        "''": "''",
        ":": ":",
        ";": ":",
        "--": ":",
        "to": "TO",
    }
    if word.lower() in known_tags_dict.keys():
        return known_tags_dict[word.lower()]

    return None


def get_history_tuple(k, x, v_tag, u_tag, t_tag):
    # next word
    n_words = len(x)
    if k == n_words - 1:  # last word
        next_word = "~"
    else:
        next_word = x[k + 1]
    # previous words
    if k == 0:
        prev_prev_word = "*"
        prev_word = "*"
    elif k == 1:
        prev_prev_word = "*"
        prev_word = x[k - 1]
    else:  # k>=2
        prev_prev_word = x[k - 2]
        prev_word = x[k - 1]

    # history tuple: (x_k, v, x_k-1, u, x_k-2, t, x_k+1)
    history_tuple = (x[k], v_tag, prev_word, u_tag, prev_prev_word, t_tag, next_word)
    return history_tuple


def get_relevant_idx(k, x, v_tag, u_tag, t_tag, histories_features, feature_to_idx):
    history_tuple = get_history_tuple(k, x, v_tag, u_tag, t_tag)
    # get relevant indexes
    if history_tuple in histories_features:
        relevant_idx = histories_features[history_tuple]
    else:
        relevant_idx = represent_input_with_features(history_tuple, feature_to_idx)
    return relevant_idx


def calc_Pi_Bp_known_tag(
    k, known_tag, B_best_idx, Pi, Bp, dict_tag_to_idx, n_tags, star_idx
):
    if k == 0:
        # u_tag = t_tag = "*"  ==>  u_idx = t_idx = star_idx
        v_idx = dict_tag_to_idx[known_tag]
        Pi[k][star_idx][v_idx] = 1
        Bp[k][star_idx][v_idx] = star_idx

    elif k == 1:
        # t_tag = "*"  ==>  t_idx = star_idx
        v_idx = dict_tag_to_idx[known_tag]
        # calc Pi, we know that (Qv / all_v will be 1)
        for u_idx in range(n_tags):
            Pi[k][u_idx][v_idx] = Pi[k - 1][star_idx][u_idx]
            Bp[k][u_idx][v_idx] = star_idx

    else:  # k>=2
        v_idx = dict_tag_to_idx[known_tag]
        # calc Pi, we know that (Qv / all_v will be 1)
        for u_idx, t_list_by_u in B_best_idx.items():
            t_scores = np.zeros(n_tags)
            for t_idx in t_list_by_u:
                t_scores[t_idx] = Pi[k - 1][t_idx][u_idx]
            Pi[k][u_idx][v_idx] = np.max(t_scores)
            Bp[k][u_idx][v_idx] = np.argmax(t_scores)

    return Pi, Bp


def calc_Q(
    k,
    x,
    Pi,
    Bp,
    tags_list,
    B_best_idx,
    histories_features,
    feature_to_idx,
    weights,
):
    n_tags = len(tags_list)
    if k == 0:
        Qv = np.zeros(n_tags)
        Q_all_v_tags = 0
        for v_idx, v_tag in enumerate(tags_list):
            relevant_idx = get_relevant_idx(
                k=k,
                x=x,
                v_tag=v_tag,
                u_tag="*",
                t_tag="*",
                histories_features=histories_features,
                feature_to_idx=feature_to_idx,
            )
            Qv[v_idx] = np.exp(weights[relevant_idx].sum())
            Q_all_v_tags += Qv[v_idx]

    elif k == 1:
        Qv = np.zeros([n_tags, n_tags])
        Q_all_v_tags = np.zeros(n_tags)
        for u_idx, u_tag in enumerate(tags_list):
            for v_idx, v_tag in enumerate(tags_list):
                relevant_idx = get_relevant_idx(
                    k=k,
                    x=x,
                    v_tag=v_tag,
                    u_tag=u_tag,
                    t_tag="*",
                    histories_features=histories_features,
                    feature_to_idx=feature_to_idx,
                )
                Qv[u_idx][v_idx] = np.exp(weights[relevant_idx].sum())
                Q_all_v_tags[u_idx] += Qv[u_idx][v_idx]

    else:  # k>=2
        Qv = np.zeros([n_tags, n_tags, n_tags])
        Q_all_v_tags = np.zeros([n_tags, n_tags])
        for u_idx, t_list_by_u in B_best_idx.items():
            u_tag = tags_list[u_idx]
            for t_idx in t_list_by_u:
                t_tag = tags_list[t_idx]
                for v_idx, v_tag in enumerate(tags_list):
                    relevant_idx = get_relevant_idx(
                        k=k,
                        x=x,
                        v_tag=v_tag,
                        u_tag=u_tag,
                        t_tag=t_tag,
                        histories_features=histories_features,
                        feature_to_idx=feature_to_idx,
                    )
                    Qv[t_idx][u_idx][v_idx] = np.exp(weights[relevant_idx].sum())
                    Q_all_v_tags[t_idx][u_idx] += Qv[t_idx][u_idx][v_idx]

    return Qv, Q_all_v_tags


def calc_Pi_Bp(k, Pi, Bp, Qv, Q_all_v_tags, tags_list, B_best_idx, star_idx):
    # TODO: check that n_tags is correct
    n_tags = len(tags_list)
    if k == 0:
        # by definition:
        # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
        # Pi(0, "*", v) = Pi(-1, "*", "*") * Q(k_history)
        # --> Pi(0, "*", v) = 1* Q(k_history)
        for v_idx, v_tag in enumerate(tags_list):
            Pi[k][star_idx][v_idx] = Qv[v_idx] / Q_all_v_tags
            Bp[k][star_idx][v_idx] = star_idx

    elif k == 1:
        # by definition:
        # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
        # Pi(1, u, v) = Pi(0, "*", u) * Q(k_history)
        for u_idx, u_tag in enumerate(tags_list):
            for v_idx, v_tag in enumerate(tags_list):
                Pi[k][u_idx][v_idx] = (
                    Pi[k - 1][star_idx][u_idx] * Qv[u_idx][v_idx] / Q_all_v_tags[u_idx]
                )
                Bp[k][u_idx][v_idx] = star_idx

    else:  # k>=2
        # by definition:
        # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
        for u_idx, t_list_by_u in B_best_idx.items():
            for v_idx, v_tag in enumerate(tags_list):
                t_scores = np.zeros(n_tags)
                for t_idx in t_list_by_u:
                    t_scores[t_idx] = (
                        Pi[k - 1][t_idx][u_idx]
                        * Qv[t_idx][u_idx][v_idx]
                        / Q_all_v_tags[t_idx][u_idx]
                    )
                Pi[k][u_idx][v_idx] = np.max(t_scores)
                Bp[k][u_idx][v_idx] = np.argmax(t_scores)

    return Pi, Bp


def memm_viterbi(sentence, pre_trained_weights, feature2id, true_tags=None):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    """
    @ v_tag: tag of current word           (position k)
    @ u_tag: tag of previous word          (position k-1)
    @ t_tag: tag of previous-previous word (position k-2)
    @ x: sentence
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
    n_words = len(x)

    B = 5  # Beam search parameter
    Pi = np.zeros([n_words, n_tags, n_tags])
    Bp = np.ones([n_words, n_tags, n_tags]) * -1
    pred_tags_idx = [-1 for _ in range(n_words)]

    dict_tag_to_idx = {v_tag: v_idx for v_idx, v_tag in enumerate(tags_list)}
    dict_idx_to_tag = {v_idx: v_tag for v_idx, v_tag in enumerate(tags_list)}
    star_idx = n_tags - 1  # "*"
    dict_tag_to_idx["*"] = star_idx
    dict_idx_to_tag[star_idx] = "*"

    feature_to_idx = feature2id.feature_to_idx
    weights = np.array(pre_trained_weights)
    histories_features = (
        feature2id.histories_features
    )  # dict histories: features relevant idx

    for k in range(n_words):
        if k == 0 or k == 1:
            B_best_idx = None
        # check if known word
        known_tag = check_if_known_word(x[k])

        if known_tag:
            Pi, Bp = calc_Pi_Bp_known_tag(
                k, known_tag, B_best_idx, Pi, Bp, dict_tag_to_idx, n_tags, star_idx
            )
        else:
            # calc Q
            Qv, Q_all_v_tags = calc_Q(
                k,
                x,
                Pi,
                Bp,
                tags_list,
                B_best_idx,
                histories_features,
                feature_to_idx,
                weights,
            )
            # calc Pi, Bp
            Pi, Bp = calc_Pi_Bp(
                k, Pi, Bp, Qv, Q_all_v_tags, tags_list, B_best_idx, star_idx
            )
        # find Top B Pi values indexes
        if k >= 1:
            B_best_idx = get_top_B_idx_dict(Pi[k], B)

    # ------------------ Get Predicted Tags by Bp ----------------------
    # last 2 words in sentence
    pred_tag_minus2_idx, pred_tag_minus1_idx = np.unravel_index(
        Pi[n_words - 1].argmax(), Pi[n_words - 1].shape
    )
    pred_tags_idx[n_words - 1] = pred_tag_minus1_idx
    pred_tags_idx[n_words - 2] = pred_tag_minus2_idx
    for k in reversed(range(n_words - 2)):
        pred_tags_idx[k] = int(Bp[k + 2][pred_tags_idx[k + 1]][pred_tags_idx[k + 2]])

    pred_tags = [dict_idx_to_tag[pred_idx] for pred_idx in pred_tags_idx]
    return pred_tags


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
        if tagged:
            true_tags = sen[1]
        else:
            true_tags = None
        # get pred
        pred_tags = memm_viterbi(sentence, pre_trained_weights, feature2id, true_tags)
        # remove * * ~
        sentence = sentence[2:-1]
        # write preds to file
        n_words = len(sentence)
        for i in range(n_words):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred_tags[i]}")
        output_file.write("\n")

        # add details tp fp ...
        if tagged:
            true_tags = sen[1][2:-1]
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
        accuracy = np.round(sum(list(Tp.values())) / n_preds, 3)

        # precision, recall, f1 for each tag
        for tag in tags_list:
            if tag == "*" or tag == "~":
                continue
            if (Tp[tag] + Fp[tag]) == 0:
                # TODO: maybe somthing else?
                precision[tag] = 0
            else:
                precision[tag] = np.round(Tp[tag] / (Tp[tag] + Fp[tag]), 2)
            if (Tp[tag] + Fn[tag]) == 0:
                recall[tag] = 1
            else:
                recall[tag] = np.round(Tp[tag] / (Tp[tag] + Fn[tag]), 2)
            if (precision[tag] + recall[tag]) == 0:
                f1[tag] = 0
            else:
                f1[tag] = np.round(
                    2 * (precision[tag] * recall[tag]) / (precision[tag] + recall[tag]),
                    2,
                )

        # f1
        mean_f1 = np.round(np.mean(list(f1.values())), 3)
        median_f1 = np.round(np.median(list(f1.values())), 3)
        print(f"{accuracy=}, {mean_f1=}, {median_f1=}")

        print("----------------------------------")
        print("10 tags with the lowest precision")
        print(sorted(precision.items(), key=lambda item: item[1])[:10])
        print("----------------------------------")
        print("10 tags with the lowest recall")
        print(sorted(recall.items(), key=lambda item: item[1])[:10])
        print("----------------------------------")
        print("10 tags with the lowest f1")
        print(sorted(f1.items(), key=lambda item: item[1])[:10])
        print("----------------------------------")

        # ----------------
        # confusion matrix
        # ----------------

        # tag_to_idx_dict = {tag: idx for (idx, tag) in enumerate(tags_list)}
        # true_list_nums = [tag_to_idx_dict[true_tag] for true_tag in true_list]
        # pred_list_nums = [tag_to_idx_dict[pred_tag] for pred_tag in pred_list]
        # n_tags = len(tags_list)
        # n_preds = len(pred_list_nums)

        # conf_matrix = np.zeros([n_tags, n_tags])
        # for i in range(n_preds):
        #     true = true_list_nums[i]
        #     pred = pred_list_nums[i]
        #     conf_matrix[true][pred] += 1

        # preds_df = pd.DataFrame(pred_list)
        # true_df = pd.DataFrame(true_list)
        # preds_df.to_csv("preds_df.csv")
        # true_df.to_csv("true_df.csv")

        # matrix_df = pd.DataFrame(conf_matrix)
        # matrix_df["tag"] = tags_list
        # a = tags_list
        # a.append(0)
        # matrix_df.loc[len(matrix_df)] = a
        # matrix_df.to_csv("conf_matrix.csv")

        print("FINISH")
        print("---")
