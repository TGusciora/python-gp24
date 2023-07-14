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

    def __init__(self, data, variable):
        """
        Constructor method
        """
        self.data = data
        self.variable = variable

    def _datetime(self):
        """
        Creating monh & day variables from datetime variable.
        """
        self.data["month"] = self.data[self.variable].dt.month
        self.data["day"] = self.data[self.variable].dt.day
        self.data["weekday"] = self.data[self.variable].dt.weekday

    def _dummy(self):
        """
        Creating dummy variables for months and days to inspect seasonality.
        """
        pd.get_dummies(data=self.data["month"], prefix="month", prefix_sep="_")
        pd.get_dummies(data=self.data["weekday"], prefix="weekday",
                       prefix_sep="_")

    def _circle(self):
        """
        Feature engineering - representing datetime variables a cyclic
        coordinates. Creating sine and cosine representation.
        """
        # Convert 'Month' and 'DayOfWeek' to cyclic coordinates
        self.data['Month_sin'] = np.sin(2*np.pi*self.data['Month']/12)
        self.data['Month_cos'] = np.cos(2*np.pi*self.data['Month']/12)
        self.data['DayOfWeek_sin'] = np.sin(2*np.pi*self.data['DayOfWeek']/7)
        self.data['DayOfWeek_cos'] = np.cos(2*np.pi*self.data['DayOfWeek']/7)
