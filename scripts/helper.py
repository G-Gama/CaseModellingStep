from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class SeasonalCols(BaseEstimator, TransformerMixin):
    """
    I'll create a value that indicates if the transaction ocurred too late 
    at night or during the weekend. First I was thinking about creating
    flags that indicated if the transaction was on the weekend or during
    the night but to be honest I see no reason to not use the full value.
    
    Here we could create stuff like "month" or "day of month" but since
    we don't have much of a historic dataframe I'll keep myself to just
    the hour and the day-of-week. 
    """
    def __init__(self, date_col):
        self.date_col = date_col
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy()
        X_[self.date_col] = pd.to_datetime(X_[self.date_col])
        X_[self.date_col+'_weekday'] = pd.to_datetime(X_[self.date_col]).dt.weekday
        X_[self.date_col+'_hour'] = pd.to_datetime(X_[self.date_col]).dt.hour
        print("DayOfWeek and Hour Variables Created!")
        
        return X_
    

    

class DatesToDateDiff(BaseEstimator, TransformerMixin):
    """
    This will help us transform dates to numbers, by taking how many days 
    passed between one event and the other one.
    """
    def __init__(self, date_cols):
        self.date_cols = date_cols
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy()
        date_cols = self.date_cols.copy()
        
        for date_col in date_cols:
            X_[date_col] = pd.to_datetime(X_[date_col])
            
        while len(date_cols) > 1:
            base_date = date_cols.pop()
            for date_col in date_cols:
                col_name = 'diff_{}_{}'.format(base_date, date_col)
                X_[col_name] = (X_[base_date] - X_[date_col])/np.timedelta64(1, 'D')
            
            print("DateDiffs regarding {} timestamp created!".format(base_date))
                
        X_.drop(columns=self.date_cols, inplace=True)
        return X_
    
    

    
class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    With this personalized OneHotEncoder, we'll map every category that has more than x%
    presence on the dataset and, if it has, create a flag regarding it. If not, we could
    call it "others" and group them, but I'll just assume it's the "leave one out" class 
    of the encoder.
    """
    def __init__(self, cat_variable, minor_category=0.05):
        self.cat_variable = cat_variable
        self.minor_category = minor_category
        
    def fit(self, X, y = None):
        X_ = X.copy()
        mapper = dict(X_[self.cat_variable].fillna('NULL').value_counts(normalize=True))
        mapper = [key for key, val in mapper.items() if val > self.minor_category]
        self.mapper = mapper
        return self

    def transform(self, X, y = None):
        X_ = X.copy()
        
        for category in self.mapper:
            cat_flag = '{}_{}'.format(self.cat_variable, category)
            X_[cat_flag] = 0
            X_.loc[X_[self.cat_variable].fillna('NULL') == category, cat_flag] = 1
        X_.drop(columns=[self.cat_variable], inplace=True)
        
        print("OneHotEncode for var {} Done!".format(self.cat_variable))
        return X_
    
    

    
class FillNulls(BaseEstimator, TransformerMixin):
    """
    Fill nulls with Special Value. Since I intend to use tree based algorithms, 
    I'll fill nulls with -999 as a special value.
    
    Kind of reinventing the wheel but atm the docs of the scikit-learn are 
    down, so I'll use custom functions.
    """
    def __init__(self, special_value=-999):
        self.special_value = special_value
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = X.copy()
        X_ = X_.fillna(self.special_value)
        print("Nulls Filled With {}!".format(self.special_value))
        return X_
    

    

class BorutaRemover(BaseEstimator, TransformerMixin):
    """
    Just remove columns, but I'll print that it happens because 
    Boruta returned them as irrelevant.
    """
    def __init__(self, remove_me):
        self.remove_me = remove_me
        
        
    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        
        X_.drop(columns=self.remove_me, inplace=True)
        vars_removed = len(self.remove_me)
        
        print('{} vars removed due to low impact by the Boruta Step!'.format(vars_removed))
        
        return X_