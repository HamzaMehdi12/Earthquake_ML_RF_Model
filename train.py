import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix

from dataset import Data_Processing

class Model_RF:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Init method for the class Model_RF
        Parameters:
            - self.X,y (both train and test) = Processed dataframe and splitted with 80-20.
            - model = RandomForestClassifier(with classifier trees 30 and random state of 20)
        Returns: 
            - Initialized states
        """
        try:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.model = RandomForestClassifier(n_estimators = 30, random_state = 20)
            self.run_pipeline()
        except Exception as e:
            print(f"Error while initializing function: {str(e)}")
            raise Exception(e)

    def run_pipeline(self):
        """
        Runs the pipeline for training, evaluating and returning results 
        """
        print("Running trainnig on the processed dataset: \n")
        self.train()
        time.sleep(3)
        print("Evaluating the results: \n")
        self.evaluate()
        time.sleep(3)
        print("saving the file: \n")
        self.save_model()

    def train(self):
        """
        Train the model. We have already defined our classifier, just need to fit.
        Parameters: 
            - self.y_pred: Prediction scores. Here:
                -- self.model.predict(X_test): Predicting on X_test case
            - self.y_prob: Probability of event occuring. Here
                -- self.model.predict_proba(X_test): Probability on X_test
            - self.y_pred_new: New predictions on unseen data
            - self.y_prob_new: New probabilities on unseen data
        Return:
            - self.y_predict
            - self.y_prob
        """
        try:
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
            return self.y_pred, self.y_prob
        except Exception as e:
            print(f"Error during training {str(e)}")
            raise Exception(e)
    
    def evaluate(self):
        """
        Evaluating the results for general realization of the model and the classifier
        Parameters:
            - Classification_report: Shows the main classification metrics 
            - ROC_AUC_SCORE: How well the model separates classes to all threshold
            - Average Precision Score: How many cases were predicted true
            - confusion matrix: What percentage of cases are real vs false
        """
        try:
            print("Classifcation report:\n ", classification_report(self.y_test, self.y_pred))
            print("ROC AUC:\t ", roc_auc_score(self.y_test, self.y_prob))
            print("Average PR AUC:\t ", average_precision_score(self.y_test, self.y_prob))

            #Now priting from the latest dataset value, when can an earthquake occur. It is region free, and does not account where can it occur. In the next update, I will add the system for location based values and changes
            print("Now testing model performance on unseen data: \n")
            time.sleep(1)
            self.y_pred_new = self.model.predict(self.X_test)
            self.y_prob_new = self.model.predict_proba(self.X_test)[:, 1]
        
            print("classification_report: \n", classification_report(self.y_test, self.y_pred_new))
            print("Confusion Matrix: \n", confusion_matrix(self.y_test, self.y_pred_new))
            print("ROC AUC", roc_auc_score(self.y_test, self.y_prob_new))

            print("Printing the results: \n")
            time.sleep(1)
            plt.figure(figsize = (10,4))
            plt.plot(self.y_prob, label = 'Predicted Probability Of Earthquake')
            plt.plot(self.y_test.values, label = 'Actual Event (1 = Quake)', alpha = 0.5)
            plt.legend()
            plt.title('Predicted Earthquale Probability vs Actual Events')
            plt.xlabel('Time Step')
            plt.ylabel('Probability/Event')
            plt.show()
            #saving image
            print("saving the result in image: \n")
            plt.savefig("Predicted_Earthquake_Probability_vs_Actual_Events.png", dpi = 300)



            #feature importance
            print("printing feature importance: \n")
            time.sleep(2)
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize = (12,6))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), np.array(self.X_train.columns)[indices], rotation = 90)
            plt.tight_layout()
            plt.show()
            #saving images
            print("saving the result in image: \n")
            plt.savefig("Feature_Importance.png", dpi = 300)
        except Exception as e:
            print(f"Error during evaluation {str(e)}")
            raise Exception(e)
    def save_model(self, filename="rf_model.pkl"):
        joblib.dump(self.model, filename)

    def load_model(self, filename="rf_model.pkl"):
        self.model = joblib.load(filename)

if __name__ == "__main__":
    try:
        df = pd.read_csv(r"earthquakes.csv", index_col = "id", usecols=['id', 'magnitude', 'type', 'title', 'date', 'time', 'felt', 'cdi', 'mmi', 'alert', 'status', 'tsunami', 'sig', 'gap', 'geometryType', 'depth', 'latitude', 'longitude', 'distanceKM', 'location', 'continent', 'country', 'subnational', 'city', 'locality', 'postcode', 'timezone'])
        print(f"visualizing the dataset: \n{df.head()}")
        process = Data_Processing(df)
        print("Processing the dataset splits")
        time.sleep(1)
        target_col = "has_eq"
        X_train, X_test, y_train, y_test = process.split(target_col, test_size = 0.2)
        print(f"Shape of training set X: {X_train.shape}, y: {y_train.shape}")
        print(f"Shape of test set X: {X_test.shape}, y: {y_test.shape}")
        
        print("Lets train and test our model: \n")
        time.sleep(2)
        Train = Model_RF(X_train, X_test, y_train, y_test)
        print("Model successfully ran..... Processing Completed\n")
        print("Shutting progam in 3 seconds:\n")

        time.sleep(3)

    except Exception as e:
        print(f"Error occured during computations: {str(e)}")
        raise Exception(e)

#------------------------------------------------------------ This completes the train file -------------------------------------------------------