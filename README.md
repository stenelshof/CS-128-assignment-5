# Assignment 5
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Presentation Link


## Author Information
* **Name(s)**: Silas Ten Elshof
* **Email(s)**: stenelshof@westmont.edu

## License Information
MIT

## How to Utilize this Software
The classifier_models file consists of several abstract classes that are utilized at other points in the code

The local_classifier_models consists of implementations of these classes. A build method is defined which builds feature sets based off of the words in the title of the source object. The gamma method can be called to give the predicted class for a given object. The present features method returns the top n most informative features for classifying. The present_features_string method is mainly for testing purposes. The train method takes in training data and calculates the counts and probabilities used by the gamma method.

The local_classifier_models_tests consists of three unit tests for each of the above mentioned methods.

The search_runner can be used to run all of these methods on actual data (I am using my YouTube search-history data from google takeout in JSON format, other formats may not be compatible with the way the code is written). It uses a train/test split and outputs an accuracy score for the classifier as well as the most informative features for classifying what year a particular search is from.

The watch_runner can be used to run all of these methods on actual data (I am using my YouTube watch-history data from google takeout in JSON format, other formats may not be compatible with the way the code is written). It uses a train/test split and outputs an accuracy score for the classifier as well as the most informative features for classifying what year a particular video watch is from.

## Citations
classifier_models code from Mike Ryu in Assignment 4

local_classifier_models from code written by myself and Luke Rozinskas for Assignment 4 (with changes to the build method)

local_classifier_modles_tests from code written by myself and Luke Rozinskas for Assignment 4 (with changes to the build tests)

search_runner and watch_runner code generated in part by ChatGBT using Mike Ryu's runner code from Assignment 4 under the following prompts:
Change this runner to take in JSON data rather than a pickle file
Change this runner so that the classes being classified are the different years from the time section in this data
