import pickle
import time

from inference import tag_all_test
from optimization import get_optimal_vector
from preprocessing import preprocess_train


def main():

    # ---- Files Paths  -----------
    weights_1_path = "weights_1.pkl"
    comp_1_input_path = "data/comp1.words"
    comp_1_output_path = "data/comp_m1_206014482_316375872.wtag"

    weights_2_path = "weights_2.pkl"
    comp_2_input_path = "data/comp2.words"
    comp_2_output_path = "data/comp_m2_206014482_316375872.wtag"

    # -----------------------------
    # Generate Comp Tagged
    # -----------------------------

    def generate_comp(weights_path, test_path, predictions_path):
        start = time.time()
        with open(weights_path, "rb") as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        print(pre_trained_weights)
        tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
        end = time.time()
        model1_time = end - start
        print(f"model test + evaluate time: {model1_time}")

    generate_comp(weights_1_path, comp_1_input_path, comp_1_output_path)
    generate_comp(weights_2_path, comp_2_input_path, comp_2_output_path)


if __name__ == "__main__":
    main()

