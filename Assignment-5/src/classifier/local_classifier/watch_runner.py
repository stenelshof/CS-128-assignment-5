import json
from random import shuffle
from src.classifier.local_classifier.local_classifier_models import LocalAbstractClassifier, LocalFeatureSet
from datetime import datetime

__author__ = "Silas Ten Elshof"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Silas Ten Elshof", "Mike Ryu"]
__license__ = "MIT"
__email__ = "Stenelshof@westmont.edu"


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    feature_sets = []
    for item in data:
        title = item.get("title", "")
        description = item.get("description", "")
        time = item.get("time", "")

        try:
            # Try to parse with the expected format
            year = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").year if time else None
        except ValueError:
            try:
                # Try to parse with an alternative format
                year = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ").year if time else None
            except ValueError:
                # Handle cases where neither format works
                print(f"Invalid timestamp: {time}")
                year = None

        feature_set = LocalFeatureSet.build([title, description], known_clas=str(year))
        feature_sets.append(feature_set)

    return feature_sets




def main():
    # Load your JSON data
    json_data = load_json_data('/Users/silastenelshof/Desktop/CS128/CS-128-assignment-5/Assignment-5/data/watch-history.json')

    # Shuffle the data
    shuffle(json_data)

    # Split the data into training and testing sets
    train_size = int(0.95 * len(json_data))
    train_data, test_data = json_data[:train_size], json_data[train_size:]

    # Train the classifier
    classifier = LocalAbstractClassifier.train(train_data)

    # Evaluate the classifier on the test set
    correct = 0
    for feature_set in test_data:
        predicted_label = classifier.gamma(feature_set)
        true_label = feature_set.clas
        if predicted_label == true_label:
            correct += 1

    # Get the accuracy of the classifier on the test data
    accuracy = correct / len(test_data)
    print(f"Classifier Accuracy: {accuracy:.2%}")

    # Present the top features
    classifier.present_features(top_n=1000)


if __name__ == "__main__":
    main()