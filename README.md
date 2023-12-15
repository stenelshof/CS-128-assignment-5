# Assignment 5
**Westmont College Fall 2023**

**CS 128 Information Retrieval and Big Data**

*Assistant Professor* Mike Ryu (mryu@westmont.edu) 

## Presentation Link
https://docs.google.com/presentation/d/1VKid3KqhRCCmL_1tKAzYqy7qDi4jQZwZKbNUIIP11t8/edit#slide=id.p

## Author Information
* **Name(s)**: Silas Ten Elshof
* **Email(s)**: stenelshof@westmont.edu

## License Information
MIT (see the LICENSE file for more specifics)

## Structure
The majority of this project's structure can be found within the Assignment-5 folder. Most of the code is located within the src/classifier folder and then within the local_classifier folder. The unittests are located just within the src/classifier. Aspects like the .gitignore and requirements.txt are within the Assignment-5 folder. The README and LICENSE are on the very outside.

## How to Utilize this Software
Within the src/classifier folder:

The classifier_models file consists of several abstract classes that are utilized at other points in the code

The local_classifier_models consists of implementations of these classes. A build method is defined which builds feature sets based off of the words in the title of the source object. The gamma method can be called to give the predicted class for a given object. The present features method returns the top n most informative features for classifying. The present_features_string method is mainly for testing purposes. The train method takes in training data and calculates the counts and probabilities used by the gamma method.

The local_classifier_models_tests consists of three unit tests for each of the above mentioned methods.

The search_runner can be used to run all of these methods on actual data (I am using my YouTube search-history data from google takeout in JSON format, other formats may not be compatible with the way the code is written). It uses a train/test split and outputs an accuracy score for the classifier as well as the most informative features for classifying what year a particular search is from.

The watch_runner can be used to run all of these methods on actual data (I am using my YouTube watch-history data from google takeout in JSON format, other formats may not be compatible with the way the code is written). It uses a train/test split and outputs an accuracy score for the classifier as well as the most informative features for classifying what year a particular video watch is from.

In order utilize this, download your JSON data to your local machine. Choose which runner you are going to use and replace the file path in the first line of the main function with the file path to your JSON data. Then you can run the runner and see the results!

## Citations
local_classifier_models and local_classifier_models_tests code written with help from Luke Rozinskas and Mike Ryu

search_runner and watch_runner code generated in part by ChatGBT using Mike Ryu's runner code from Assignment 4 under the following prompts:
Change this runner to take in JSON data rather than a pickle file
Change this runner so that the classes being classified are the different years from the time section in this data
