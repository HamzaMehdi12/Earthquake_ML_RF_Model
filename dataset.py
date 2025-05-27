#dataset.py
#Cleaning, ordering, encoding, preprocessing and visualizing the data

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



class Data_Processing:
    def __init__(self, df):
        """
        Dunder method self calls for the class. OOP based calling with class operator self.
        """
        print("Initializing the process and processing")
        self.df = df
        self.run_pipeline()
        
    def run_pipeline(self):
        """
        Running the pipeline for automated processing. Update with jobluib will be served in phase 2
        """
        print("Preprocessig the dataset")
        self.df = self.preprocessing()
        time.sleep(3)
        print("Visualizing the dataset")
        self.visualization()
        time.sleep(3)
        
    def visualization(self):
        """
        Visualize and read the parameters for data validatoin and preprocessing
        Parameters:
        - self.df: pandas DataFrame to process
        - time: to sleep for the next dataset to be processed
        - 
        Returns:
        - Processed DataFrame
            -- plt.title(): Adds title to plot
            -- plt.tight_layout: Adjusts the paddng to ensure fit of plot
            -- plt.show(): shows the graph
        - Plotting for values for earthquake in a certain area. Here:
            -- plotting on 20, 8 figzie
            -- Combining Area from city, subnational and country, using astype(str) + ','(using seperater)
            -- plotting bar using ax.bar()
            --saving image on computer with the name    
        - If failed, returns -1
        
        """
        try: 
            #viewig the results
            self.df.drop(columns = ['has_eq'])  
            if self.df is None:
                raise ValueError('Data not processed')
            print(f"Data loaded successfully! First five rows:\n ", self.df.head())
            print(f"Printing the info of data with datatypes: \n")
            self.df.info()
            
            #plotting results of dataset before PCA 
            plt.figure(figsize = (10,4))
            self.df[self.target_name].plot()
            plt.title('Daily Earthquake Occurance')
            plt.xlabel('Date')
            plt.ylabel('Earthquake')
            plt.tight_layout()
            plt.show()
            #saving image
            print("saving the result in image: \n")
            plt.savefig("Daily_Earthquake_Occurance.png", dpi = 300)


            #using Principle Component Analysis (PCA).
            #preparing for pca
            numeric_df = self.df.select_dtypes(include = [np.number]).dropna(axis = 1, how = 'all')
            if numeric_df.shape[1] > 2:
                pca = PCA(n_components = 2)
                reduced = pca.fit_transform(self.df.drop(columns = [self.target_name]))
                plt.figure(figsize = (8,6))
                plt.scatter(reduced[:,0], reduced[:,1], alpha = 0.6)
                plt.title('PCA of processed features')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.tight_layout()
                plt.show()
                #saving image
                print("saving the result in image: \n")
                plt.savefig("PCA.png", dpi = 300)
            else:
                raise ValueError("Not enough numeric features for PCA")
            
            #saving file column names
            print("Writing the columns in file: \n")
            with open("columns_names.txt", "w", encoding = "utf-8") as f:
                for col in self.df.columns:
                    f.write(col + "\n")
        except Exception as e: 
            print(f"Error received during printing and plotting: {str(e)}")
            raise Exception(e)

    def preprocessing(self):

        """
        Preprocessed and encodes datafeatures for further computations.
        Parameters:
            - self.df: Dataframes
            - self.df.isnull().any().any: Views whether am entry is null or not
            - pd.DataFrame() used to return bac to pandas framework after imputing.
            - imp: A simple imputer for checking null values. Here:
                -- num_imp and cat_imp for numerical and catageories
                -- self.df: dataframe
                -- using numeric and categorical transformers
                -- missing_values = np.nan: Values to be targeted
                -- strategy = mean: calculates mean of the column and replaces the value here
            - enc: OneHotEncoder for encoding textual values to integer values. Here:
                -- df_enc: The dataframe encoded
                -- fit_transform: fit the dataframe
                -- cols: selets the columns to encode. Here:
                    --- include = ['object']: Selects only string values
                    --- columns: select the columns
                    --- tolist(): make a list of the columns selected
                -- self.df[cols]: dataframe including only textual columns
                -- pd.Dataframe: creates a Dataframe of the encoded columns. Here:
                    --- one_enc: The encoded columns
                    --- columns = enc.get_feature_names_out(cols): names of columns encoded
            - Column Transformers: Transforming the dataset back to the original dataset, while dropping other values
            - Pipeline: For pipeline based function

            - Sort values: Sorts the values from oldest to latest
            - Feature Engineering:
                -- self.df['hour] - featured engineered with hour column and indexed with self.df.index.hour (0-23)
                -- self.df['days of the week'] - feature engineered with weeks column and indexed with self.df.index.daysofweek
                -- self.df['month'] - feature engineered with months and oindexed with self.df.index.month
            - Aggregate values for daily values using agg = self.df.resample('D').agg({
                'magnitude': 'mean',
                'depth': 'mean',
                'latitude': 'mean'
                'longitude': 'mean'
                }).fillna(0)
                    -- fillna(0) fills where no data is present
            - df_ts = daily.join(agg), joins daily targets into DF
            - Lag Features: binary targets for time values for 1, 7, 14 days
        Returns:
            - self.df: Dataframe returned
        """
        try:
            #parsing and indexing time and date
            self.df['datetime'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('datetime').set_index('datetime')

            #dropping unnecesaary cols
            self.df.drop(columns = ['date', 'time', 'title', 'location'], inplace = True)

            #Feature Engineering
            self.df['hour'] = self.df.index.hour
            self.df['dayofweek'] = self.df.index.dayofweek
            self.df['month'] = self.df.index.month

            #Generating daily targets: Earthquake occurance
            daily = self.df.resample('D').size().to_frame('quake_count')
            daily['has_eq'] = (daily['quake_count'] > 0).astype(int)

            #Aggregate Features daily
            agg = self.df.resample('D').agg({
                'magnitude': 'mean',
                'depth' : 'mean',
                'latitude' : 'mean',
                'longitude' : 'mean'
                }).fillna(0)

            #Combining features with target
            self.df_ts = daily.join(agg)

            #Creating lag features for sequence context
            for lag in [1, 7, 14]:
                self.df_ts[f'lag_{lag}_occ'] = self.df_ts['has_eq'].shift(lag)
            self.df_ts.dropna(inplace=True)
            self.target_name = 'has_eq'
            X = self.df_ts.drop(columns = [self.target_name])
            y = self.df_ts[self.target_name]
            print(f"Sample size: X {X.shape}, y: {y.shape}")
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()
            #defining imputers
            num_imp = SimpleImputer(strategy='mean')
            cat_imp = SimpleImputer(strategy = 'constant', fill_value = 'missing')

            #imputing, scaling and encoding both num cat columns
            numeric_transform = Pipeline(steps = [
                ('imputer', num_imp),
                ('scaler', RobustScaler())
                ])
            print(f"After imputing, dataset: {self.df.head()}")
            transformers = [('num', numeric_transform, num_cols)]
            if cat_cols:
                categorical_transform = Pipeline(steps = [
                    ('imputer', cat_imp),
                    ('encoder', OneHotEncoder(handle_unknown = 'ignore', sparse_output=False))
                    ])
                transformers.append(('cat', categorical_transform, cat_cols))
            #combining using Column Transformers
            self.preprocessor = ColumnTransformer(transformers)
            X_processed = self.preprocessor.fit_transform(X)
            #extractig names
            feat_names = list(num_cols)

            if cat_cols:
                encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
                cat_feats = encoder.get_feature_names_out(cat_cols)
                feat_names += list(cat_feats)

            #Building processed DataFrame
            self.df = pd.DataFrame(X_processed, index = X.index, columns = feat_names)
            self.df[self.target_name] = y.loc[self.df.index]
            #saving file to csv
            self.df.to_csv("processed_earthquake_data.txt", sep = '\t', index=True, encoding='utf-8')
            return self.df
        except Exception as e:
            print(f"Error eccured during imputing: {str(e)}")
            raise Exception(e)


    def split(self, target_col, test_size = 0.2):
        """
        Splits the dataset into train and test datasets.
        Parameters:
            - X = self.df.drop(): Dataframe without the target column. The rest remains the same
            - y = self.df.dtop[target]: Only the target column

            - X_train, y_train: Splitted training dataset of X and y
            - X_test, y_test: Splitted training dataset of X and y

            - train_test_split(X,y,test_size, random_state: Sciket learn splitting function. Here;
                -- Test_size: size of test to be used
                -- random_state: shuffling and random rows.
                -- stratify: To help imbalances in class

        Returns:
            - X_train, y_train
            - X_test, y_test
        """
        try:
            X = self.df.drop(columns = [target_col])
            y = self.df[target_col]
            print(f"Sample values: X: {X.shape}, y: {y.shape}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 20, stratify = y)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error received durig splitting: {str(e)}")
            raise Exception(e)

#--------------------------------------------------------- This completes the dataset file --------------------------------------------------------