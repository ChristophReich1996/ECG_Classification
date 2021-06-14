from random import shuffle

from ecg_classification import TRAINING_SPLIT_CHALLANGE
from wettbewerb import load_references

if __name__ == '__main__':
    # Get data
    ecg_leads, ecg_labels, fs, ecg_names = load_references("../data/training2017/")
    # Get indexes of class normal and AF
    indexes = [index for index, label in enumerate(ecg_labels) if label in ["N", "A"]]
    # Shuffle indexes
    shuffle(indexes)
    # Get training indexes
    training_indexes = [index for index in indexes if index in TRAINING_SPLIT_CHALLANGE]
    # Get possible validation indexes
    validation_indexes_possible = [index for index in indexes if index not in TRAINING_SPLIT_CHALLANGE]
    # Update training indexes
    training_indexes.extend(validation_indexes_possible[300:])
    validation_indexes = validation_indexes_possible[:300]
    print(training_indexes)
    print(validation_indexes)
