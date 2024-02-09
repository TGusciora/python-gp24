import pandas as pd
import ta
import numpy as np
import datetime
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


class DateFeatures:
    """
    Class creating date variables from datetime variable.

    Creates dummy variables for months and days to inspect seasonality.
    Furthermore, creates cycle representation on a circle.

    Parameters
    -----------
    data : str
        Dataset to which date variables will be added.
    variable : str
        Base datetime variable to do feature engineering on.

    Attributes
    ----------
    data_out : pandas DataFrame
        Created pandas Dataframe with transformed variables.

    Notes
    ---------------------
    Required libraries: \n
    * import pandas as pd \n
    * import datetime \n

    Methods
    -------
    __init__(self, file_name)
        Constructor method.
    _datetime(self)
        Creates datetime variables.
    _dummy(self)
        Creates dummy variables for months and days to inspect seasonality.
    _circle(self)
        Creates mont and weekday variables circle representations.
    output(self)
        Returns transformed pandas DataFrame.
    """

    def __init__(self, data_in, variable):
        """
        Constructor method
        """
        self.data_in = data_in
        self.variable = variable
        self.data_out = pd.DataFrame()
        self.data_out = data_in.copy()
        self._datetime()
        self._dummy()
        self._circle()

    def output(self):
        """
        Generate transformed output.
        """
        return self.data_out

    def _datetime(self):
        """
        Creating monh & day variables from datetime variable.
        """
        self.data_out["month"] = self.data_in[self.variable].dt.month
        self.data_out["day"] = self.data_in[self.variable].dt.day
        self.data_out["weekday"] = self.data_in[self.variable].dt.weekday

    def _dummy(self):
        """
        Creating dummy variables for months and days to inspect seasonality.
        """
        pd.get_dummies(data=self.data_out["month"],
                       prefix="month", prefix_sep="_")
        pd.get_dummies(data=self.data_out["weekday"],
                       prefix="weekday", prefix_sep="_")
        self._datetime()  # restore month & weekday variables

    def _circle(self):
        """
        Feature engineering - representing datetime variables a cyclic
        coordinates. Creating sine and cosine representation.
        """
        self.data_out['month_sin'] = np.sin(2*np.pi*self.data_out['month']/12)
        self.data_out['month_cos'] = np.cos(2*np.pi*self.data_out['month']/12)
        self.data_out['weekday_sin'] = np.sin(2*np.pi*self.data_out['weekday']/7)
        self.data_out['weekday_cos'] = np.cos(2*np.pi*self.data_out['weekday']/7)


class FeaturesOneHotEncoding:
    """
    Class creating new features from pointed variables using OneHotEncoding.

    Transforms pointed variables to binary variables and returns transformed
    DataFrame.

    Parameters
    -----------
    data_in : str
        Dataset on which encoder training will be performed.
    features_list : str
        Name of list storing features names and their values for encoder input.

    Attributes
    ----------
    data_transformed : pandas DataFrame
        Created pandas Dataframe with transformed variables using trained
        OneHotEncoder.

    Notes
    -------------------
    Imported raw file should be a .csv file. \n
    Required libraries: \n
    * import pandas as pd \n
    * from sklearn.preprocessing import OneHotEncoder
    * from sklearn.compose import make_column_transformer

    Methods
    -------
    __init__(self, file_name)
        Constructor method.
    _datetime(self)
        Creates datetime variables.
    _dummy(self)
        Creates dummy variables for months and days to inspect seasonality.
    _circle(self)
        Creates mont and weekday variables circle representations.
    output(self)
        Returns transformed pandas DataFrame.
    """

    def __init__(self, data_in, features_list):
        """
        Constructor method.
        """
        self.data_in = data_in
        self.features_list = features_list
        self._init_params()
        self._init_encoder()
        self._init_transformer()

    def _init_params(self):
        """
        Creating parameters for encoder and column transformed from 
        features_list.
        """
        self.ohe_columns = [x[0] for x in self.features_list]
        self.ohe_categories = [x[1] for x in self.features_list]
        return self

    def _init_encoder(self):
        """
        Sklearn OneHotEncoder construction.

        Using categories from features_list to create OneHotEncoder instance.
        """
        self.enc = OneHotEncoder(sparse_output=False,
                                 categories=self.ohe_categories)
        return self.enc

    def _init_transformer(self):
        """
        Sklearn column transformer construction.

        Using columns from features_list and OneHotEncoder to create column
        transforer. Fitting transformer on data_in instance.
        """
        self.transformer = make_column_transformer((self.enc, self.ohe_columns))
        self.transformer.fit_transform(self.data_in)
        return self.transformer

    def transform(self, data_to_transform):
        """
        Sklearn column transformer construction.

        Using columns from features_list and OneHotEncoder to create column
        transforer. Fitting transformer on data_in instance. Removing
        "onehotencoder__" prefix from column names.
        """
        self.data_to_transform = data_to_transform
        self.data_transformed = pd.DataFrame(
            self.transformer.transform(self.data_to_transform),
            columns=self.transformer.get_feature_names_out(),
            index=self.data_to_transform.index
            )
        # strip suffix
        self.data_transformed.columns = self.data_transformed.columns.str.removeprefix("onehotencoder__")
        return self.data_transformed


class VarDescriptiveStatistics:
    """
    Class creating new features from variable using descriptive statistics.

    Takes numeric variable and provides based on given list of days - mean,
    standard deviation, maximum and minimum values over last X days (where
    X is a given number from list of days). Furthermore creates ratios between
    calculated variables (for example ratio of last X days mean to last Y days
    mean).

    Parameters
    -----------
    data_in : str
        Dataset on which features will be created.
    days : str
        Name of list storing days numbers for calculating features and ratios.
    var_name : str
        Name of base variable for feature engineering.
    shift : int, default 1
        Indicator if output needs to be shifted before return (calculated 
        values moved 1 observations forward)

    Attributes
    ----------
    results : pandas DataFrame
        Resulting dataframe with features generated.

    Notes
    -------------------
    Imported raw file should be a .csv file. \n
    Required libraries: \n
    * import pandas as pd \n
    * import numpy as np \n
    * import itertools

    Methods
    -------
    __init__(self, file_name)
        Constructor method.
    _comibnations(self)
        Creates datetime variables.
    _dummy(self)
        Creates dummy variables for months and days to inspect seasonality.
    _circle(self)
        Creates mont and weekday variables circle representations.
    output(self)
        Returns transformed pandas DataFrame.
    """

    def __init__(self, data_in, days, var_name, shift = 1):
        """
        Constructor method.
        """
        self.data_in = data_in
        self.days = days
        self.var_name = var_name
        self.shift = shift
        self.res = {}  # results dictionary
        self._combinations()
        self._stats()
        self._ratios()

    def output(self):
        """
        Generate transformed output.

        Converts res dictionary to results pandas DataFrame. Returns
        DataFrame without original variables from data_in.
        """
        self.results = pd.DataFrame(self.res)

        if self.shift == 1:
            # Shifting 1 observation, because we don't want to use current
            # observation target information for calculation (data leak).
            self.results = self.results.shift(periods=1)
        return self.results

    def _combinations(self):
        """
        Creating combinations of days numbers.

        Using combinations with replacement on sorted list of days.
        """
        self.days.sort()
        self.combinations = list(itertools.combinations_with_replacement(self.days, 2))

    def _stats(self):
        """
        Creating descriptive statistics for numeric variables.

        For each day from days list creates last X days mean, stdev, max and
        min.
        """
        for i in self.days:
            self.res[f'{self.var_name}_mean_{i}'] = self.data_in[self.var_name].rolling(i).mean()
            self.res[f'{self.var_name}_stdev_{i}'] = self.data_in[self.var_name].rolling(i).std()
            self.res[f'{self.var_name}_max_{i}'] = self.data_in[self.var_name].rolling(i).max()
            self.res[f'{self.var_name}_min_{i}'] = self.data_in[self.var_name].rolling(i).min()

    def _ratios(self):
        """
        Creating ratios between calculated variables.

        For each descriptive statistic from _stats function creates a ratio
        between variables. Creates ratio only for same var_name variable
        transformation.
        """
        for combo in self.combinations:
            d1, d2 = combo
            if d2 >= d1:
                mean_stdev_ratio = np.where(self.res[f'{self.var_name}_stdev_{d2}'] != 0,
                                            self.res[f'{self.var_name}_mean_{d1}'] / self.res[f'{self.var_name}_stdev_{d2}'],
                                            np.nan)
                mean_max_ratio = np.where(self.res[f'{self.var_name}_max_{d2}'] != 0,
                                          self.res[f'{self.var_name}_mean_{d1}'] / self.res[f'{self.var_name}_max_{d2}'],
                                          np.nan)
                mean_min_ratio = np.where(self.res[f'{self.var_name}_min_{d2}'] != 0,
                                          self.res[f'{self.var_name}_mean_{d1}'] / self.res[f'{self.var_name}_min_{d2}'],
                                          np.nan)
                stdev_mean_ratio = np.where(self.res[f'{self.var_name}_mean_{d2}'] != 0,
                                            self.res[f'{self.var_name}_stdev_{d1}'] / self.res[f'{self.var_name}_mean_{d2}'],
                                            np.nan)
                stdev_max_ratio = np.where(self.res[f'{self.var_name}_max_{d2}'] != 0,
                                           self.res[f'{self.var_name}_stdev_{d1}'] / self.res[f'{self.var_name}_max_{d2}'],
                                           np.nan)
                stdev_min_ratio = np.where(self.res[f'{self.var_name}_min_{d2}'] != 0,
                                           self.res[f'{self.var_name}_stdev_{d1}'] / self.res[f'{self.var_name}_min_{d2}'],
                                           np.nan)
                max_mean_ratio = np.where(self.res[f'{self.var_name}_mean_{d2}'] != 0,
                                         self.res[f'{self.var_name}_max_{d1}'] / self.res[f'{self.var_name}_mean_{d2}'],
                                         np.nan)
                max_min_ratio = np.where(self.res[f'{self.var_name}_min_{d2}'] != 0,
                                         self.res[f'{self.var_name}_max_{d1}'] / self.res[f'{self.var_name}_min_{d2}'],
                                         np.nan)
                max_stdev_ratio = np.where(self.res[f'{self.var_name}_stdev_{d2}'] != 0,
                                         self.res[f'{self.var_name}_max_{d1}'] / self.res[f'{self.var_name}_stdev_{d2}'],
                                         np.nan)
                min_mean_ratio = np.where(self.res[f'{self.var_name}_mean_{d2}'] != 0,
                                         self.res[f'{self.var_name}_min_{d1}'] / self.res[f'{self.var_name}_mean_{d2}'],
                                         np.nan)
                min_max_ratio = np.where(self.res[f'{self.var_name}_max_{d2}'] != 0,
                                         self.res[f'{self.var_name}_min_{d1}'] / self.res[f'{self.var_name}_max_{d2}'],
                                         np.nan)
                min_stdev_ratio = np.where(self.res[f'{self.var_name}_stdev_{d2}'] != 0,
                                         self.res[f'{self.var_name}_min_{d1}'] / self.res[f'{self.var_name}_stdev_{d2}'],
                                         np.nan)

                # assign to dictionary
                self.res[f'{self.var_name}_mean_{d1}_stdev_{d2}_ratio'] = mean_stdev_ratio
                self.res[f'{self.var_name}_mean_{d1}_max_{d2}_ratio'] = mean_max_ratio
                self.res[f'{self.var_name}_mean_{d1}_min_{d2}_ratio'] = mean_min_ratio
                self.res[f'{self.var_name}_stdev_{d1}_mean_{d2}_ratio'] = stdev_mean_ratio
                self.res[f'{self.var_name}_stdev_{d1}_max_{d2}_ratio'] = stdev_max_ratio
                self.res[f'{self.var_name}_stdev_{d1}_min_{d2}_ratio'] = stdev_min_ratio
                self.res[f'{self.var_name}_max_{d1}_mean_{d2}_ratio'] = max_mean_ratio
                self.res[f'{self.var_name}_max_{d1}_stdev_{d2}_ratio'] = max_stdev_ratio
                self.res[f'{self.var_name}_max_{d1}_min_{d2}_ratio'] = max_min_ratio
                self.res[f'{self.var_name}_min_{d1}_mean_{d2}_ratio'] = min_mean_ratio
                self.res[f'{self.var_name}_min_{d1}_stdev_{d2}_ratio'] = min_stdev_ratio
                self.res[f'{self.var_name}_min_{d1}_max_{d2}_ratio'] = min_max_ratio

                if d2 > d1:
                    # calculate variable value
                    mean_ratio = np.where(self.res[f'{self.var_name}_mean_{d2}'] != 0,
                                          self.res[f'{self.var_name}_mean_{d1}'] / self.res[f'{self.var_name}_mean_{d2}'],
                                          np.nan)
                    stdev_ratio = np.where(self.res[f'{self.var_name}_stdev_{d2}'] != 0,
                                           self.res[f'{self.var_name}_stdev_{d1}'] / self.res[f'{self.var_name}_stdev_{d2}'],
                                           np.nan)
                    max_ratio = np.where(self.res[f'{self.var_name}_max_{d2}'] != 0,
                                         self.res[f'{self.var_name}_max_{d1}'] / self.res[f'{self.var_name}_max_{d2}'],
                                         np.nan)
                    min_ratio = np.where(self.res[f'{self.var_name}_min_{d2}'] != 0,
                                         self.res[f'{self.var_name}_min_{d1}'] / self.res[f'{self.var_name}_min_{d2}'],
                                         np.nan)
                    # assign to dictionary
                    self.res[f'{self.var_name}_mean_{d1}_mean_{d2}_ratio'] = mean_ratio
                    self.res[f'{self.var_name}_stdev_{d1}_stdev_{d2}_ratio'] = stdev_ratio
                    self.res[f'{self.var_name}_max_{d1}_max_{d2}_ratio'] = max_ratio
                    self.res[f'{self.var_name}_min_{d1}_min_{d2}_ratio'] = min_ratio
        return self.res
