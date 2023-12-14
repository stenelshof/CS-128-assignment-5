"""Implementation of a Naive Bayes Text Classifier"""


from nltk import word_tokenize
from nltk.corpus import stopwords
from src.classifier.classifier_models import *


__author__ = "Silas Ten Elshof"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Silas Ten Elshof", "Mike Ryu"]
__license__ = "MIT"
__email__ = "Stenelshof@westmont.edu"

class LocalFeature(Feature):
    def __init__(self, name, value=None):
        super().__init__(name, value)

class LocalFeatureSet(FeatureSet):
    @classmethod
    def build(cls, source_object: Any, known_clas=None, remove_stopwords=True, min_word_length=5,
              **kwargs) -> FeatureSet:
        title = source_object[0]  # Extract the title from the source_object

        # Define stopwords for the title
        stop_words = set(stopwords.words('english'))

        # Create the LocalFeatureSet with some preprocessing applied to the title, take out shorter words and numbers
        features = LocalFeatureSet({
            LocalFeature(word.lower(), value=True)
            for word in word_tokenize(title)
            if len(word) >= min_word_length and word.isalpha() and (
                    not remove_stopwords or word.lower() not in stop_words)
        }, known_clas)

        return features


class LocalAbstractClassifier(AbstractClassifier):

    def __init__(self, feature_probabilities: dict, clas_probabilities: dict):
        self.feature_probabilities = feature_probabilities
        self.clas_probabilities = clas_probabilities

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified, returns the most probable class
                for the object based on the training this classifier received (via a call to `train` class method).
                :param a_feature_set: a single feature set representing an object to be classified
                :return: name of the class with the highest probability for the object
                """

        # Set the predicted class variable and an empty set
        predicted_class = None
        prior_prob = {}

        # This is iterating over each class label that we have (pro or con)
        # in the probability set list that we got from train
        # it accesses the 3rd feature in our tuple which corresponds with the feature class label
        for class_label in set(feature[2] for feature in self.feature_probabilities.keys()):
            # We have to set the class probability to be multiplied against each feature to then update it with the max
            class_probability = 1.0
            # print("class label", class_label)

            # Iterate over each feature in the feature
            # set gotten from training
            for feature in a_feature_set.feat:

               # We intialize the feature key for each feature
               # in the feature set we pass in
                feature_key = (feature.name, feature.value, class_label)
                # We check if the feature key for the feature is in the probability set that we received
                if feature_key in self.feature_probabilities:
                    # if so, we multiply the probability we calculated previously with the class prob
                    class_probability *= self.feature_probabilities[feature_key]
                # Here we update the prior probability for the class with the newly assigned class probability
                prior_prob[class_label] = class_probability

        # Last we make sure there is a prior_probability and then update the class with the max value
        if prior_prob:
            predicted_class = max(prior_prob, key=prior_prob.get)

        return predicted_class

    def present_features(self, top_n: int = 1) -> None:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
                feature in determining a class for any object. Informativeness of a feature is a quantity that represents
                how "good" a feature is in determining the class for an object.

                :param top_n: how many of the top features to print; must be 1 or greater
                """
        #print(self.present_features_string(top_n))
        if top_n < 1:
            raise ValueError("top_n must be 1 or greater")

        # Collect feature informativeness
        odd_ratio_collection = {}

        sorted_features = sorted(self.feature_probabilities.items(), key=lambda item: item[0])
        # print(sorted_features)
        for index in range(0, len(sorted_features), 2):
            next_element = index + 1

            feature1_prob = sorted_features[index][1]
            first_class = sorted_features[index][0][2]
            feature2_prob = sorted_features[next_element][1]
            second_class = sorted_features[next_element][0][2]

            # feature_key, probability = sorted_features
            # feature_name, feature_value, class_label = feature_key

            if feature1_prob == 0.0:
                feature1_prob = 0.00001
            if feature2_prob == 0.0:
                feature2_prob = 0.00001
            if feature1_prob > feature2_prob:
                # odd_ratio_collection[sorted_features[index[0][0]]] = {(feature1_prob/feature2_prob), first_class}
                odd_ratio_collection[sorted_features[index][0][0]] = [(feature1_prob / feature2_prob), first_class]

            else:
                # odd_ratio_collection[sorted_features[next_element[0][0]]] = {(feature2_prob/feature1_prob), second_class}
                odd_ratio_collection[sorted_features[next_element][0][0]] = [(feature2_prob / feature1_prob), second_class]

        sorted_odds = sorted(odd_ratio_collection.items(), key=lambda item: item[1][0], reverse=True)

        # Print the top N features
        print("Most Informative Features")
        for i in range(min(top_n, len(sorted_odds))):
            print(f" ({sorted_odds[i]})")

    def present_features_string(self, top_n: int = 1) -> list[str]:
        odd_ratio_collection = {}

        sorted_features = sorted(self.feature_probabilities.items(), key=lambda item: item[0])

        for index in range(0, len(sorted_features) - 1):
            next_element = index + 1

            # By iterating over every 2 elements, we grab their probabilities for each class
            feature1_prob = sorted_features[index][1]
            first_class = sorted_features[index][0][2]
            feature2_prob = sorted_features[next_element][1]
            second_class = sorted_features[next_element][0][2]

            # Do some accounting for the divide by zero error
            if feature1_prob == 0.0:
                feature1_prob = 0.00001
            if feature2_prob == 0.0:
                feature2_prob = 0.00001
            if feature1_prob > feature2_prob:
                odd_ratio_collection[sorted_features[index][0][0]] = [(feature1_prob / feature2_prob), first_class]
            else:
                odd_ratio_collection[sorted_features[next_element][0][0]] = [(feature2_prob / feature1_prob),
                                                                             second_class]


        sorted_odds = sorted(odd_ratio_collection.items(), key=lambda item: item[1][0], reverse=True)
        result = [str(entry) for entry in sorted_odds[:top_n]]
        return result
    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        """Method that builds a Classifier instance with its training (supervised learning) already completed. That is,
                the `AbstractClassifier` instance returned as the result of invoking this method must support `gamma` and
                `present_features` method calls immediately without needing any other method invocations prior to them.

                :param training_set: An iterable collection of `FeatureSet` to use for training the classifier
                :return: an instance of `AbstractClassifier` with its training already completed
                """

        # Create counts as dictionaries to be accessed via keys
        feature_count = {}
        clas_count = {}

        # iterating through each feature set in the training set we take in
        for feat_set in training_set:
            # We make sure to set the clas that should be taken in from the
            # feature sets that are passed in, and if not its unknown
            actual_class_label = feat_set.clas if feat_set.clas else "unknown"

            # Counting for classes, incrementing and setting as necessary
            if actual_class_label in clas_count:
                clas_count[actual_class_label] += 1
            else:
                clas_count[actual_class_label] = 1

            # Counting for features, we set the feature key to the name, value,
            # and class corresponding to the class it passed in and we count like we did before
            for feature in feat_set.feat:
                feature_key = (feature.name, feature.value, feat_set.clas)
                if feature_key in feature_count:
                    feature_count[feature_key] += 1
                else:
                    feature_count[feature_key] = 1

        # Set up two dictionaries for probabilities,
        # those of the features and those of the classes
        features_probabilities = {}
        clas_probabilities = {}

        # we set a total class occurrences sum in order to calc the
        # probability of a class when looking at the features
        total_class_occurrences = sum(clas_count.values())

        # We then iterate over the feature keys in the count that we incremented above
        for feature_key, count in feature_count.items():
            feature_name, feature_value, class_label = feature_key

            # We iterate within the loop over the classes in the class count
            for possible_class in set(clas_count.keys()):
                # Here we create a feature key for the current class
                current_feature_key = (feature_name, feature_value, possible_class)

                # if the feature key we look at is in our count, then we calculate the probability
                if current_feature_key in feature_count:
                    features_probabilities[current_feature_key] = count / clas_count[possible_class]
                else:
                    # If the class doesn't have any occurrences, set probability to 0
                    features_probabilities[current_feature_key] = 0.0

        # Calculate class probabilities here using that total occurrences we had before and the individual class counts
        for class_label, class_count in clas_count.items():
            clas_probabilities[class_label] = class_count / total_class_occurrences

        return LocalAbstractClassifier(features_probabilities, clas_probabilities)

