#-------------------------------------------------------------------------
# AUTHOR: Kate Yuan
# FILENAME: decision_tree_2.py
# SPECIFICATION: tests 3 sets of trainging data on decision trees
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

# for column in dbTest.columns:
#     dbTest[column] = pd.factorize(dbTest[column])[0] + 1 

# TestX = dbTest.values
# TestY = dbTest['Recommended Lenses'].values


for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dbTraining = pd.read_csv(ds)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    featureColumns = dbTraining.columns[:-1]

    mappings = {}
    for col in featureColumns:
        categories = dbTraining[col].unique().tolist()
        mappings[col] = {cat: i+1 for i, cat in enumerate(categories)}
        dbTraining[col] = dbTraining[col].map(mappings[col])
    
    X = dbTraining[featureColumns].values

    # #Transform the original categorical training classes to numbers and add to the vector Y.
    # #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    # #--> add your Python code here
    # Y = dbTraining['Recommended Lenses'].values
    classColumn = dbTraining.columns[-1]
    classCategories = dbTraining[classColumn].unique().tolist()
    classMap = {cat: i+1 for i, cat in enumerate(classCategories)}
    dbTraining[classColumn] = dbTraining[classColumn].map(classMap)

    Y = dbTraining[classColumn].values

    accuracy = 0
    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       # clf =
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
        clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here

        correct = 0
        total = 0
        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            test_features = []
            for j, col in enumerate(featureColumns):
                test_features.append(mappings[col][data[j]])
            
            class_predicted = clf.predict([test_features])[0]
           

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            true_label = classMap[data[-1]]
            if(class_predicted == true_label):
                correct += 1
            total += 1

        testAccuracy = float(correct/total)
        accuracy += testAccuracy

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

    finalAccuracy = accuracy / 10.0

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here

    print(f"final accuracy when training on {ds}: {finalAccuracy}" )




