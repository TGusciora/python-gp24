import pandas as pd
import ta
import numpy as np
import datetime


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
    data : pandas DataFrame
        imported file
    delimiter :str
        Delimiter in imported raw file. Used as pandas read_csv parameter.

    Notes
    -------------------
    Imported raw file should be a .csv file. \n
    Required libraries: \n
    * import pandas as pd \n
    * import datetime \n

    Methods
    -------
    __init__(self, file_name)
        Constructor method.
    _import_dataset(self)
        Imports file_name and returns as self.data pandas DataFrame.
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
