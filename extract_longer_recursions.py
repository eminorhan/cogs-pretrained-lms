import argparse
import json
import os
import logging
from collections import Counter

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

# General arguments
parser.add_argument("--input_data_file", type=str,
                    default="data/gen.json", help="Path to input data file to extract examples from.")
parser.add_argument("--conditions_file", type=str,
                    default="gen_conditions.txt", help="Path to the text file with a condition specification for each "
                                                       "example in the input_data_file.")
parser.add_argument("--training_data_file", type=str,
                    default="data/train.json", help="Path to input data file to extract examples from.")
parser.add_argument("--output_data_file_pattern", type=str,
                    default="data/{}_recursion_depth_from_{}_to_{}.json",
                    help="Filename for the data with added longer conditions.")


def main(flags):
    assert os.path.exists(flags["input_data_file"]), "No input file found at %s" % flags["input_data_file"]
    assert os.path.exists(flags["conditions_file"]), "No conditions file found at %s" % flags["conditions_file"]
    assert os.path.exists(flags["training_data_file"]), "No training data file found at %s" % flags["training_data_file"]

    with open(flags["conditions_file"]) as infile:
        conditions_data = json.load(infile)

    conditions_to_extract = ["cp_recursion", "pp_recursion"]
    num_total_examples = 0
    pp_recursion_words = ["in", "on", "beside"]
    normal_train_data = []
    recursions_data = {}
    num_training_examples = 0
    with open(flags["training_data_file"]) as infile:
        for line in infile.readlines():
            num_training_examples += 1
            normal_train_data.append(json.loads(line))

    logger.info("Num training examples before extra recursions: %d" % num_training_examples)
    normal_test_data = []
    normal_test_conditions = []
    num_test_examples = 0
    with open(flags["input_data_file"]) as infile:
        for line, condition in zip(infile.readlines(), conditions_data):
            num_total_examples += 1
            input_example = json.loads(line)
            if condition in conditions_to_extract:
                input_sentence = input_example["translation"]["en"].split()
                if condition == "cp_recursion":
                    recursions = input_sentence.count("that")
                elif condition == "pp_recursion":
                    recursions = 0
                    for word in input_sentence:
                        if word in pp_recursion_words:
                            recursions += 1
                else:
                    raise ValueError("Wrong recursion condition %s" % condition)
                if recursions not in recursions_data:
                    recursions_data[recursions] = {}
                    recursions_data[recursions]["data"] = []
                    recursions_data[recursions]["conditions"] = []
                recursions_data[recursions]["data"].append(input_example)
                recursions_data[recursions]["conditions"].append(condition)
            else:
                num_test_examples += 1
                normal_test_data.append(input_example)
                normal_test_conditions.append(condition)

    logger.info("Test examples before extra recursions: %d" % num_test_examples)

    assert num_total_examples == len(conditions_data), "Mismatch in number of input examples and number of conditions. "\
                                                       " (%d versus %d)" % (num_total_examples, len(conditions_data))

    max_recursion = max(list(recursions_data.keys()))
    min_recursion = min(list(recursions_data.keys()))
    for i in range(min_recursion, max_recursion):
        extra_training_examples = []
        extra_test_examples = []
        test_conditions = normal_test_conditions.copy()
        recursions_train = [j for j in range(min_recursion, i + 1)]
        recursions_test = [j for j in range(i + 1, max_recursion + 1)]
        assert set(recursions_train).isdisjoint(set(recursions_test)), "Overlap in train and test."
        train_file = flags["output_data_file_pattern"].format("train", str(0), str(i))
        test_file = flags["output_data_file_pattern"].format("test", str(i + 1), max_recursion)
        conditions_file = flags["output_data_file_pattern"].format("conditions", str(i + 1), max_recursion)
        extra_train_conditions = []
        for rec_train in recursions_train:
            extra_training_examples.extend(recursions_data[rec_train]["data"])
            extra_train_conditions.extend(recursions_data[rec_train]["conditions"])
        for rec_test in recursions_test:
            extra_test_examples.extend(recursions_data[rec_test]["data"])
            test_conditions.extend(recursions_data[rec_test]["conditions"])
        assert Counter(extra_train_conditions + test_conditions) == Counter(conditions_data), \
            "A condition got lost."
        total_train_examples = normal_train_data + extra_training_examples
        train_str = [str(ex) for ex in total_train_examples]
        with open(train_file, "w") as outfile:
            for train_example in total_train_examples:
                json.dump(train_example, outfile)
                outfile.write('\n')
        total_test_examples = normal_test_data + extra_test_examples
        test_str = [str(ex) for ex in total_test_examples]
        assert set(train_str).isdisjoint(set(test_str)), "Overlap train and gen."
        assert len(total_test_examples) == len(test_conditions), "Incorrect conditions file."
        with open(test_file, "w") as outfile:
            for test_example in total_test_examples:
                json.dump(test_example, outfile)
                outfile.write('\n')
        with open(conditions_file, "w") as outfile:
            json.dump(test_conditions, outfile)
        logger.info("File %s and %s. %d train examples, %d test examples, %d together." % (
            train_file, test_file, len(total_train_examples), len(total_test_examples),
            len(total_train_examples) + len(total_test_examples)))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
