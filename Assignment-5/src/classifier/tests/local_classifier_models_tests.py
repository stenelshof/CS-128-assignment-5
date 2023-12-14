"""Unit tests for classes and methods for `local_classifier_models`.
"""
import unittest
from src.classifier.local_classifier.local_classifier_models import LocalFeatureSet, LocalAbstractClassifier, LocalFeature


__author__ = "Silas Ten Elshof"
__copyright__ = "Copyright 2023, Westmont College"
__credits__ = ["Silas Ten Elshof", "Mike Ryu"]
__license__ = "MIT"
__email__ = "Stenelshof@westmont.edu"


class LocalFeatureSetTest(unittest.TestCase):
    def test_build(self):
        doc11 = [{
            "header": "YouTube",
            "title": "Watched bex",
            "titleUrl": "https://www.youtube.com/watch?v\u003d6pI01HTiZ1c",
            "description": "Watched at 4:51 PM",
            "time": "2023-11-22T00:51:23.799Z",
            "products": ["YouTube"],
            "details": [{
                "name": "From Google Ads"
            }],
            "activityControls": ["Web \u0026 App Activity", "YouTube watch history", "YouTube search history"]
        }]

        # Extract the title string from the dictionary
        titles = [item.get("title", "") for item in doc11]

        build11 = LocalFeatureSet.build(source_object=titles, known_clas="2023")
        expected_features = {LocalFeature("watched", value=True)}
        self.assertEqual(expected_features, build11.feat)

        doc12 = [{
            "header": "YouTube",
            "title": "Watched bex awesome assignment",
            "titleUrl": "https://www.youtube.com/watch?v\u003d6pI01HTiZ1c",
            "description": "Watched at 4:51 PM",
            "time": "2023-11-22T00:51:23.799Z",
            "products": ["YouTube"],
            "details": [{
                "name": "From Google Ads"
            }],
            "activityControls": ["Web \u0026 App Activity", "YouTube watch history", "YouTube search history"]
        }]

        # Extract the title string from the dictionary
        titles = [item.get("title", "") for item in doc12]

        build12 = LocalFeatureSet.build(source_object=titles, known_clas="2023")
        expected_features = {LocalFeature("watched", value=True), LocalFeature("awesome", value=True), LocalFeature("assignment", value=True)}
        self.assertEqual(expected_features, build12.feat)

        doc13 = [{
            "header": "YouTube",
            "title": "hi bex and a fun guy",
            "titleUrl": "https://www.youtube.com/watch?v\u003d6pI01HTiZ1c",
            "description": "Watched at 4:51 PM",
            "time": "2023-11-22T00:51:23.799Z",
            "products": ["YouTube"],
            "details": [{
                "name": "From Google Ads"
            }],
            "activityControls": ["Web \u0026 App Activity", "YouTube watch history", "YouTube search history"]
        }]

        # Extract the title string from the dictionary
        titles = [item.get("title", "") for item in doc13]

        build13 = LocalFeatureSet.build(source_object=titles, known_clas="2023")
        expected_features = set()
        self.assertEqual(expected_features, build13.feat)


class LocalAbstractClassifierTest(unittest.TestCase):

    def test_gamma(self):
        training_set1 = LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)},
                                        known_clas="pro")
        classifier = LocalAbstractClassifier.train([training_set1])
        expected_probabilities = {("good", True, "pro"): 1.0,
                                  ("good", True, "con"): 0.0,
                                  ("great", True, "pro"): 1.0,
                                  ("great", True, "con"): 0.0,
                                  ("bad", True, "con"): 1.0,
                                  ("bad", True, "pro"): 0.0,
                                  ("horrible", True, "con"): 1.0,
                                  ("horrible", True, "pro"): 0.0}
        expected_class_probabilities = {"pro": 0.5, "con": 0.5}

        for feature_set in [training_set1]:
            prediction = classifier.gamma(feature_set)
            # print the prediction for debugging
            print(f"Predicted class: {prediction}")
            self.assertEqual(prediction, "pro")

        training_set2 = LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)},
                                        known_clas="con")
        classifier2 = LocalAbstractClassifier.train([training_set2])
        for feature_set in [training_set2]:
            prediction2 = classifier2.gamma(feature_set)
            print(f"Predicted class: {prediction2}")
            self.assertEqual(prediction2, "con")


        training_set3 = LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("good", value=True), LocalFeature("great", value=True)},
                                        known_clas="pro")
        classifier3 = LocalAbstractClassifier.train([training_set3])
        for feature_set in [training_set3]:
            prediction3 = classifier3.gamma(feature_set)
            print(f"Predicted class: {prediction3}")
            self.assertEqual(prediction3, "pro")


    def test_present_features_string(self):
        training_set1 = [
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)}, known_clas="pro"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("horrible", value=True)}, known_clas="con")]

        classifier = LocalAbstractClassifier.train(training_set1)

        top_features = classifier.present_features_string(top_n=2)
        print(top_features)
        expected_features = ["('bad', [99999.99999999999, 'con'])", "('good', [99999.99999999999, 'pro'])"]
        self.assertEqual(expected_features, top_features)

        training_set2 = [
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)}, known_clas="pro"),
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("horrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con")]

        classifier2 = LocalAbstractClassifier.train(training_set2)
        top_features2 = classifier2.present_features_string(top_n=3)
        print(top_features2)
        expected_features = ["('good', [99999.99999999999, 'pro'])", "('bad', [66666.66666666666, 'con'])", "('terrible', [66666.66666666666, 'con'])"]
        self.assertEqual(expected_features, top_features2)

        training_set3 = [
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)}, known_clas="pro"),
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("horrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con")]

        classifier3 = LocalAbstractClassifier.train(training_set3)
        top_features3 = classifier3.present_features_string(top_n=4)
        print(top_features3)
        expected_features = ["('good', [99999.99999999999, 'pro'])", "('bad', [75000.0, 'con'])", "('terrible', [75000.0, 'con'])", "('horrible', [24999.999999999996, 'con'])"]
        self.assertEqual(expected_features, top_features3)

    def test_train(self):

        training_set1 = [
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)}, known_clas="pro"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("horrible", value=True)}, known_clas="con")]

        classifier = LocalAbstractClassifier.train(training_set1)
        expected_probabilities = {("good", True, "pro"): 1.0,
                                  ("good", True, "con"): 0.0,
                                  ("great", True, "pro"): 1.0,
                                  ("great", True, "con"): 0.0,
                                  ("bad", True, "con"): 1.0,
                                  ("bad", True, "pro"): 0.0,
                                  ("horrible", True, "con"): 1.0,
                                  ("horrible", True, "pro"): 0.0}
        expected_class_probabilities = {"pro": 0.5, "con": 0.5}
        print("expected",expected_probabilities)


        print("classy", classifier.feature_probabilities)
        self.assertEqual(expected_probabilities, classifier.feature_probabilities)
        self.assertEqual(expected_class_probabilities, classifier.clas_probabilities)

        training_set2 = [
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)}, known_clas="pro"),
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("horrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con")]

        classifier2 = LocalAbstractClassifier.train(training_set2)
        expected_probabilities2 = {("good", True, "pro"): 1.0,

                                  ("great", True, "pro"): 1.0,
                                   ("great", True, "con"): 0.0,
                                  ("good", True, "con"): 0.3333333333333333,
                                  ("horrible", True, "con"): 0.3333333333333333,
                                   ("horrible", True, "pro"): 0.0,
                                   ("bad", True, "con"): 0.6666666666666666,
                                   ("bad", True, "pro"): 0.0,
                                ("terrible", True, "con"): 0.6666666666666666,
                                   ("terrible", True, "pro"): 0.0}
        expected_class_probabilities = {"pro": 0.25, "con": 0.75}

        print("expected",expected_probabilities2)


        print("classifier",classifier2.feature_probabilities)
        self.assertEquals(expected_probabilities2, classifier2.feature_probabilities)
        self.assertEqual(expected_class_probabilities, classifier2.clas_probabilities)

        training_set3 = [
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("great", value=True)}, known_clas="pro"),
            LocalFeatureSet({LocalFeature("good", value=True), LocalFeature("horrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con"),
            LocalFeatureSet({LocalFeature("bad", value=True), LocalFeature("terrible", value=True)}, known_clas="con")]


        classifier3 = LocalAbstractClassifier.train(training_set3)
        expected_probabilities3 = {("good", True, "pro"): 1.0,
                                   ("great", True, "pro"): 1.0,
                                   ("great", True, "con"): 0.0,
                                   ("good", True, "con"): 0.25,
                                   ("horrible", True, "con"): 0.25,
                                   ("horrible", True, "pro"): 0.0,
                                   ("bad", True, "con"): 0.75,
                                   ("bad", True, "pro"): 0.0,
                                   ("terrible", True, "con"): 0.75,
                                   ("terrible", True, "pro"): 0.0}
        expected_class_probabilities = {"pro": 0.2, "con": 0.8}

        self.assertEquals(expected_probabilities3, classifier3.feature_probabilities)
        self.assertEqual(expected_class_probabilities, classifier3.clas_probabilities)


if __name__ == '__main__':
    unittest.main()
