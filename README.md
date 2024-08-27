
## <span style="font-weight:bold;">Logistic Regression, Decision Tree, KNearest Neighbors, and Support Vector Machines Classifiers </span>
### <span style="font-weight:bold;">Overview</span>
This project compares the performance of classic classifiers: Logistic Regression, Decision Tree, KNearest Neighbors, and Support Vector Machines
using dataset provided by a Portuguese banking institution.  A configuration .ini file is used to selectively control the regressor used allowing 
individual debugging of each classifier model.
Two generic classification functions are defined, the first performing the basic model classification based on the regressor passed in.  The second 
function utilizes the grid hyper-parameters passed into to perform a GridSearch using the regressor.  Classification model training times, and 
the model's classification report is returned for comparasions at the end.

Classifier performance is performed by comparing Accuracy, Precision, Recall, F1-scoring, model training time from the model classification process.

### <span style="font-weight:bold;">Source:</span>  <span style="color:black;">https://archive.ics.uci.edu/dataset/222/bank+marketing</span>
The data is a "Multivariate" business use data from direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based 
on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be 
('yes') or not ('no') subscribed. This project uses the dataset: bank-addition-full.csv with 41,188 rows with 20 columns, ordered by date (from May 2008 to 
November 2010).  The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

### <span style="font-weight:bold;">Project Organization</span>
The project is organized with the objective that it to be used in an automated environment.  Individual directories, configuration files, and trained models
can be wrote out and read back for testing new data.
#### <span style="font-weight:bold;">Dataset: </span>./data/bank-additional-full.csv
#### <span style="font-weight:bold;">Configuration:</span> ./BankTermDeposit.ini
The configuration file serves multiple purposes: it identifies the source of the model training data, controls the train/test data split ratio, manages verbosity, 
and oversees the activation training and testing of the classifier. Additionally, it specifies the name of the trained model for local storage.
#### <span style="font-weight:bold;">Trained Models: </span> specified by model_outputFile in .ini
Generated for each of the classifier activated in the configuration file.  Example:  <model_outputFile>_LogisticRegression<timestamp>.pkl.
These trained model files are stored in the local directory.  These models can be read back and used for testing to classify new unknown datasets with
the same data frame format.
#### <span style="font-weight:bold;">Results</span>
Classification results from the selected classifiers are tabulated and printed at the end of the process run.






### <span style="font-weight:bold;">Process Flow</span>
#### <span style="font-weight:bold;">Configuration</span>
Read in necessary system and process flow control related configuration from the supplied .ini file.
#### <span style="font-weight:bold;">Data Pre-processing, cleaning</span>
Prepare the data by removing bad data, null values, and make the data frame available as the common dataset for later process stages.  Details are provided 
later in this project.
#### <span style="font-weight:bold;">Split data for training and testing</span>
The dataset is split according to the to proportions specified in the configuration file: BankTermDeposit.ini, variable: train_test_split, currently set to 0.3 split.

#### <span style="font-weight:bold;">Generic Classification Modules</span>
These are the 2 generic classifiers functions performing classification, and grid search based on the regressor passed in.  The resulting classification report and 
measured training time along with accuracy scores are gathered for later comparisions with other regressors.  Classifier training time, or the data fitting time is 
calculated by timestamping the start and end of the classifier training data fit time.

#### <span style="font-weight:bold;">Classification</span>
The execution of each of the regressors: Logistic Regression, Decision Tree, KNearest Neighbors, and Support Vector Machines are controlled by ini file's 
parameters: LogisticRegression, SVMGridSearch,DecisionTreeClassifier,KNNearestNeighbors.  Each module performs basic regressor model classification, and then
prepares the proper hyper-parameters for performing grid search.  The results from these classification tasks are gathered, and tabulated later for comparing
the different classifier.
#### <span style="font-weight:bold;">Results Tabulation</span>
This step tabulates the results gathered from all the classification performed.  Based on the size of the dataset, the specific trained module can be stored,
and read back directly from a file to be used directly for prediction purposes.

## <span style="font-weight:bold;">Classifier Comparison Result</span>

Basic Classifiers

|    Classifier Model    | Accuracy | Precision | Recall | F1 Score | Train Score | Test Score | Model Fit Time (s) |
|------------------------|----------|-----------|--------|----------|-------------|------------|--------------------|
|    LinearRegression    |  0.9131  |  0.9017   | 0.9131 |  0.9028  |   0.9082    |   0.9131   |     0.0550551      |
|  KNeighborsClassifier  |  0.9037  |  0.8926   | 0.9037 |  0.8962  |   0.9267    |   0.9037   |     0.0156393      |
|     SVMGridSearch      |  0.9134  |  0.9017   | 0.9134 |  0.9020  |   0.9185    |   0.9134   |     4.2046297      |
| DecisionTreeClassifier |  0.8889  |  0.8930   | 0.8889 |  0.8909  |   1.0000    |   0.8889   |     0.1003885      |


GridSearch Best Estimators

|    Classifier Model    | Accuracy | Precision | Recall | F1 Score | Train Score | Test Score | Model Fit Time (s) |                         hyper-parameters                         |
|------------------------|----------|-----------|--------|----------|-------------|------------|--------------------|------------------------------------------------------------------|
|    LinearRegression    |  0.9131  |  0.9017   | 0.9131 |  0.9028  |   0.9081    |   0.9131   |     6.2736654      |      {'regressor__C': 1, 'regressor__solver': 'liblinear'}       |
|  KNeighborsClassifier  |  0.9075  |  0.8934   | 0.9075 |  0.8948  |   1.0000    |   0.9075   |     52.1640265     | {'regressor__n_neighbors': 20, 'regressor__weights': 'distance'} |
|     SVMGridSearch      |  0.9118  |  0.8994   | 0.9118 |  0.8983  |   0.9130    |   0.9118   |    1197.6210067    |          {'regressor__C': 10, 'regressor__gamma': 0.01}          |
| DecisionTreeClassifier |  0.9127  |  0.9094   | 0.9127 |  0.9109  |   0.9345    |   0.9127   |     7.6676013      | {'regressor__max_depth': 10, 'regressor__min_samples_split': 10} |

##### Given the same dataset, all classifiers showed similar accuracy and precision perform, with similar training and testing score.
##### For this set of banking data, the Logistic, and SVM classifier showing relatively close results showing the data classification favors linear type classification
##### Classifiers SVM, and KNearest Neighbors favoring non-linear classification have bad timing performance yielding similar accuracy performance
