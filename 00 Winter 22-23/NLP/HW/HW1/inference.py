import numpy as np
from tqdm import tqdm

from preprocessing import read_test


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    """
    @ n: number of words in the sentence
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
    # TODO: add n_tags, tags_list to feature2id class
    tags_list = feature2id.tags_list  # list of all possible tags
    n_tags = feature2id.n_tags  # number of tags in train set
    x = sentence
    weights = np.array(pre_trained_weights)
    n_words = len(x)
    Pi = np.zeros([n_words, n_tags, n_tags])
    Bp = np.zeros([n_words, n_tags, n_tags])
    pred_tags = np.array(n_words)

    # histories_features: OrderedDict[history_tuple: [relevant_features_indexes]]
    histories_features = feature2id.histories_features

    for k in range(n_words):

        # ------------------ Calc Pi k = 0 ----------------------
        if k == 0:
            # by defenition:
            # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
            # Pi(0, "*", v) = Pi(-1, "*", "*") * Q(k_history)
            # --> Pi(0, "*", v) = 1* Q(k_history)

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
                Pi[k]["*"][v_idx] = Qv[v_idx] / Q_all_v_tags

        # ------------------ Calc Pi k = 1 ----------------------
        if k == 1:
            # by defenition:
            # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)
            # Pi(1, u, v) = Pi(0, "*", u) * Q(k_history)

            # calculate Q and we'll use it later a lot
            Qv = np.zeros([n_tags, n_tags])
            Q_all_v_tags = np.zeros(n_tags)
            for u_idx, u_tag in enumerate(tags_list):
                for v_idx, v_tag in enumerate(tags_list):
                    # history tuple: (x_k, v, x_k-1, u, x_k-2, t, x_k+1)
                    history_tuple = (x[k], v_tag, x[k - 1], u_tag, "*", "*", x[k + 1])
                    relevant_idx = histories_features[history_tuple]
                    Qv[u_idx][v_idx] = np.exp(weights[relevant_idx].sum())
                    Q_all_v_tags[u_idx] += Qv[u_idx][v_idx]

            # calculate Pi
            for u_idx, u_tag in enumerate(tags_list):
                for v_idx, v_tag in enumerate(tags_list):
                    Pi[k][u_idx][v_idx] = (
                        Pi[k - 1]["*"][u_idx] * Qv[u_idx][v_idx] / Q_all_v_tags[u_idx]
                    )

        # ------------------ Calc Pi k > 1 ----------------------
        else:
            # by defenition:
            # Pi(k, u, v) = Pi(k-1, t, u) * Q(k_history)

            # calculate Q and we'll use it later a lot
            Qv = np.zeros([n_tags, n_tags, n_tags])
            Q_all_v_tags = np.zeros([n_tags, n_tags])
            for t_idx, t_tag in enumerate(tags_list):
                for u_idx, u_tag in enumerate(tags_list):
                    for v_idx, v_tag in enumerate(tags_list):
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
            for u_idx, u_tag in enumerate(tags_list):
                for v_idx, v_tag in enumerate(tags_list):
                    t_scores = np.zeros(n_tags)
                    for t_idx, t_tag in enumerate(tags_list):
                        t_scores[t_idx] = (
                            Pi[k - 1][t_idx][u_idx]
                            * Qv[t_idx][u_idx][v_idx]
                            / Q_all_v_tags[t_idx][u_idx]
                        )
                    Pi[k][u_idx][v_idx] = np.max(t_scores)
                    Bp[k][u_idx][v_idx] = tags_list[np.argmax(t_scores)]

    # ------------------ Get Predicted Tags by Bp ----------------------
    # last 2 words in sentence
    pred_tags[n_words - 1] = tags_list[np.argmax(Pi[n_words - 1])]
    pred_tags[n_words - 2] = tags_list[np.argmax(Pi[n_words - 2])]
    for k in reversed(range(n_words - 2)):
        pred_tags[k] = Bp(k + 2, pred_tags[k + 1], pred_tags[k + 2])

    return pred_tags


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
