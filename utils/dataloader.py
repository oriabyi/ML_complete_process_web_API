import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # fill Nan with median
        for col in ['total eve minutes', 'total eve charge', 'total intl minutes', 'total intl calls']:
            self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mean())
            
        # fill Nan with mode
        self.dataset['number vmail messages'] = self.dataset['number vmail messages'].fillna(self.dataset['number vmail messages'].mode()[0])

        # replace value
        self.dataset['called_csc_more_2'] = 0
        self.dataset.loc[self.dataset['customer service calls'] > 2, 'called_csc_more_2'] = 1

        # columns combination
        self.dataset['total_calls'] = self.dataset['total day calls'] + self.dataset['total eve calls'] + self.dataset['total night calls'] + self.dataset['total intl calls']
        self.dataset['total_minutes'] = self.dataset['total day minutes'] + self.dataset['total eve minutes'] + self.dataset['total night minutes'] + self.dataset['total intl minutes']
        self.dataset['total_charge'] = self.dataset['total day charge'] + self.dataset['total eve charge'] + self.dataset['total night charge'] + self.dataset['total intl charge']

        # binning with q/cut
#         self.dataset['total_charge'] = pd.qcut(self.dataset['total_charge'], 4)
#         self.dataset['total_calls'] = pd.qcut(self.dataset['total_calls'], 4)
#         self.dataset['total_minutes'] = pd.qcut(self.dataset['total_minutes'], 4)
#         self.dataset['number vmail messages'] = pd.cut(self.dataset['number vmail messages'], 5)


#         # encode labels
        le = LabelEncoder()
        self.dataset['state'] = le.fit_transform(self.dataset['state'])
        self.dataset['number vmail messages'] = le.fit_transform(self.dataset['number vmail messages'])
        self.dataset['churn'] = le.fit_transform(self.dataset['churn'])
        self.dataset['voice mail plan'] = le.fit_transform(self.dataset['voice mail plan'])
        self.dataset['international plan'] = le.fit_transform(self.dataset['international plan'])

        # one hot encode
#         ohe = OneHotEncoder()
        ohe_cols = ['area code']#, 'number vmail messages', 'total_charge', 'total_calls', 'total_minutes']
#         q = ohe.fit_transform(self.dataset[ohe_cols]).toarray()
        for i in range(len(ohe_cols)):
            self.dataset = pd.concat([self.dataset, pd.get_dummies(self.dataset[ohe_cols[i]], prefix='category')], axis=1)

#         q_cols = [['category_'+str(el) for el in self.dataset[col].unique()] for col in ohe_cols]
#         q_cols_flatten = [item for sublist in q_cols for item in sublist]

        self.dataset = self.dataset.drop(ohe_cols+['id'], axis=1)

#         all_cols = np.hstack((self.dataset.columns, q_cols_flatten))
#         self.dataset = pd.DataFrame(np.hstack((self.dataset.values, q)), columns=all_cols, dtype=np.float64)

        return self.dataset
