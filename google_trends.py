from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY, WEEKLY
from typing import Tuple

import pandas as pd
import sqlalchemy as db
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import models
import assistant_functions as af
from dynamic_time_warping import DynamicTimeWarping


class Query:
    """
    Class Query holds in an object-type a query, the town that it's based on, its rescaled values based on a combination
    of regression models we passed as parameters and its training period, its aggregated data from the combination of
    regression models used and average data based on the weights of each regression model
    """
    _default = 7  # number of days that we will use for the timedelta operations later on
    _steps = [1, 2, 3, 4, 21, 26]  # number of steps that will skip in the date loop
    _frequencies = [i * 7 for i in _steps]  # number of days that correspond to 1, 2, 3, 4, etc. weeks

    def __init__(self,
                 query: str = None,
                 hashed_query: str = None,
                 start: datetime = datetime(2015, 1, 1),  # random starting date
                 end: datetime = datetime.today(),
                 days: int = _default,
                 origin: str = 'GR',
                 scheduler: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        query:
            str
        hashed_query:
            str
        start:
            date
        end:
            date
        origin:
            str
        scheduler:
            bool - flag that indicates whether or not the class is called by the weekly scheduler in order to update the
            data
        """
        print(query, hashed_query, start, end, days, origin)
        self._query = query[0]
        self._language_code = query[1]
        if hashed_query:
            self._hashed_query = hashed_query
        else:
            self._hashed_query = af.create_hash(self._query)
        self._destination = self._query.split()[1]
        self._start = start
        self._end = end
        self._origin = origin
        self._scheduler = scheduler
        self._autoencoder = None
        self._rescaler = models.LSTM(**kwargs)

        self._database = af.retrieve_scaled(self._hashed_query, self._destination)
        if self._database is None:
            raw = af.retrieve_raw(self._hashed_query, self._destination)
            raw = self.denoise(raw)
            self._database = self.rescale(raw)
        else:
            if self._database.index[-1] is not datetime.today().day:
                self._database, self._raw_data = self.update(self._database, days=self._default)

        t_f = self._start.strftime('%Y-%m-%d') + ' ' + self._end.strftime('%Y-%m-%d')
        self._timeline = af.fetch_dataframes(query, t_f, location=self._origin)
        self._dtw = ErrorWrapper.error(self._database, self._timeline, case='dtw')

        if not self._scheduler:
            self.store_data()

    @staticmethod
    def _create_df(initial_df, data):
        """Auxiliary function that takes the data that relate to the initial_df and creates the corresponding dataframe"""
        data = data.reshape(data.shape[0], data.shape[1])
        return pd.DataFrame(data=data, index=initial_df.index, columns=[initial_df.keys()[0]])

    def denoise(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Function that takes the raw data and removes noise outliers that affect the trend"""
        new_data = pd.DataFrame()

        d1 = self._start
        d2 = calculate_d2(d1, self._end)
        data = dataframe.loc[d1:d2].values

        self._autoencoder = models.LstmAE(self._default, 1)
        self._autoencoder.compile(loss='mae', optimizer='adam')
        reshaped_data = data.reshape(data.shape[0], 1, data.shape[1])
        _ = self._autoencoder.fit(reshaped_data, reshaped_data, epochs=50, verbose=1, shuffle=False)
        denoised_data = self._autoencoder.predict(reshaped_data)

        new_data = new_data.append(self._create_df(dataframe.loc[d1:d2], denoised_data))

        for dt in rrule(freq=WEEKLY, dtstart=d2+timedelta(days=1), until=self._end):
            d2 = calculate_d2(dt, self._end)
            data = dataframe.loc[dt:d2].values
            reshaped_data = data.reshape(data.shape[0], 1, data.shape[1])
            denoised_data = self._autoencoder.predict(reshaped_data)

            new_data = new_data.append(self._create_df(dataframe.loc[d1:d2], denoised_data))

        return pd.DataFrame(data=new_data, index=dataframe.index, columns=[dataframe.keys()[0]])

    def rescale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates uniformly scaled Google Trends data.

        A TensorFlow Neural Network model rescales the raw Google Trends data that are fetched and stored in such way
        directly from Google. Each frequency that the data are stored and fetched have different scales based on the
        number of searches that a query had for that period of time.

        In order to eliminate this "Google scale" we train Neural Network models that do this exact job. In order to be
        as accurate as it can be the raw Google Trends data are fetched and stored in many different frequencies in
        order to ensure the different multiple scales that Google has for a certain date in different time frames.

        Lastly, the models reset after a certain period of time in order to ensure and avoid an "overfitting trend",
        along with that each different period of time has different a different trend, e.g. a summer specified search
        has a greater trend during the spring period since people are search for something to do in the summer that is
        to come.

        Parameters
        ----------
        data:
            DataFrame - contains the raw Google Trends data

        Returns
        -------
        database:
            pd.DataFrame - a pandas DataFrame that contains the rescaled Google Trends data
        """
        interval = calculate_interval(self._default)

        database = pd.DataFrame()

        for month_start in rrule(freq=MONTHLY, dtstart=self._start, interval=interval, until=self._end):
            month_end = calculate_month_end(month_start, self._end, interval)

            first_week_end_dt = month_start + timedelta(days=self._default - 1)  # end of first week
            first_week = data.loc[month_start:first_week_end_dt]
            database = database.append(first_week)

            second_week_start_dt = first_week_end_dt + timedelta(days=1)
            second_week_end_dt = calculate_auxiliary_date(second_week_start_dt, self._default, self._end)
            second_week = data.loc[second_week_start_dt:second_week_end_dt]
            t_f = month_start.strftime('%Y-%m-%d') + ' ' + second_week_end_dt.strftime('%Y-%m-%d')

            auxiliary_table = af.fetch_dataframes(self._query, t_f, location=self._origin)

            self._rescaler.sampling(second_week, auxiliary_table, first_week)
            self._rescaler.build()
            self._rescaler.train()

            scaled = self._rescaler(second_week)

            database = database.append(scaled)

            aux_start = second_week_end_dt + timedelta(days=1)

            if month_end > self._end:
                month_end = self._end
            for current_date in rrule(freq=WEEKLY, dtstart=aux_start, until=month_end):
                d1 = current_date
                d2 = calculate_d2(d1, month_end)
                next_week = data.loc[d1:d2]
                scaled = self._rescaler(next_week)
                database = database.append(scaled)

        return database

    def update(self,
               data: pd.DataFrame,
               end: datetime = datetime.today(),
               days: int = _default) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Updates until an end date the uniformly scaled Google Trends data.

        Function that appends into the "data" DataFrame the newly fetched Google Trends data, rescales them to a uniform
        scale and stores both the scaled and the raw data.

        Parameters
        ----------
        data:
            DataFrame - outdated data that is already stored in the database and needs to be updated with data until
            the today's or a different date.
        end:
            date - a date that indicates the end date of the update for the data, default is set to today's date,
            though a different day is also possible.
        days:
            int - number of days indicating the frequency in which the incoming data will be, it may vary as well

        Returns
        -------
        tuple:
            data: a pandas DataFrame that contains the updated uniformly scaled data.
            raw: a pandas DataFrame that contains the raw data from Google Trends itself along with the updated data
            until end's date.
        """
        print('i am inside the update function')
        raw_data = self._raw_data
        start = data.index[-1]
        interval = calculate_interval(days)
        self._end = end

        for month_start in rrule(freq=MONTHLY, dtstart=start, interval=interval, until=end):
            month_end = calculate_month_end(month_start, end, interval)

            first_week_start_dt = month_start - timedelta(days=days - 1)  # end of first week
            first_week = data.loc[first_week_start_dt:month_start]

            second_week_start_dt = (month_start + timedelta(days=1)).strftime('%Y-%m-%d')
            second_week_end_dt = (second_week_start_dt + timedelta(days=days - 1)).strftime('%Y-%m-%d')
            second_week = af.fetch_dataframes(self._query, second_week_start_dt + ' ' + second_week_end_dt,
                                              self._origin)

            auxiliary_table = af.fetch_dataframes(self._query, first_week_start_dt.strftime('%Y-%m-%d') + ' ' +
                                                  second_week_end_dt, location=self._origin)
            lstm = models.LSTM()
            lstm.sampling(second_week, auxiliary_table, first_week)
            lstm.build()
            lstm.train()
            scaled = lstm(second_week)
            data = data.append(scaled)

            aux_start = second_week_end_dt + timedelta(days=1)
            for current_date in rrule(freq=WEEKLY, dtstart=aux_start, until=month_end):
                d1 = current_date
                d2 = calculate_d2(d1, month_end)

                next_week = data.loc[d1:d2]
                scaled = lstm(next_week)
                data = data.append(scaled)

        return data, raw_data

    def store_data(self) -> None:
        """Stores Google Trends data in a database.

        Stores the uniformly scaled Google Trends data into a database and identifies the correct hashed_query that
        corresponds to the correct query that we have collected the data.

        The flag is used whether or not is storing new query with scaled data or it's updating the data until today's
        date or some other date that is passed as a parameter. Along with the update of the scaled data the missing raw
        Google Trends data are also stored in the database and are updated and identified with the same way.
        """
        engine, connection, metadata = af.create_connection()

        df_json = self._database.to_json()
        scaled = metadata.tables['aihg_scaled_trends']
        # if flag is True then stores new data
        if not self._scheduler:
            insert = db.insert(scaled).values(
                hash=self._hashed_query, start=self._start.date(), end=self._end.date(), origin=self._origin,
                language_code=self._language_code, destination=self._destination, data=df_json
            )
            _ = connection.execute(insert)
        # else if flag is False then updates the already stored data with the new that we fetched
        else:
            update = db.update(scaled).where(scaled.c.hash == self._hashed_query).values(end=self._end.date(),
                                                                                         data=df_json)
            _ = connection.execute(update)

        af.close_connection(engine, connection)

    @property
    def database(self):
        return self._database

    @property
    def raw_data(self):
        return self._raw_data


class ErrorWrapper:
    """
    Class that calls upon the error functions to calculate between 2 time series, in our case the isolated/raw data
    with some scaled data
    """

    def r2(*time_series):
        """r^2 error calculator"""
        return r2_score(*time_series)

    def dtw(*time_series, bag_size=60):
        """Dynamic Time Warping error calculator"""
        DTW = DynamicTimeWarping()
        DTW.get(time_series[0], time_series[1], bucket_size=bag_size)
        return DTW.D

    def pearson(*time_series):
        """Pearson' correlation coefficient calculator"""
        return pearsonr(*time_series)

    error_functions = {'r2': r2,
                       'dtw': dtw,
                       'pearson': pearson}

    @staticmethod
    def error(time_series1, time_series2, case='dtw'):
        """Returns a specified error for 2 time series

        Parameters
        ----------
        time_series1:
            nparray - contains the scaled or raw data
        time_series2:
            nparray - contains the scaled or raw data
        case:
            string - contains the method that will calculate the error or correlation between the pair of data
        """
        func = ErrorWrapper.error_functions.get(case)
        ts1 = time_series1.values.reshape((1, -1))
        ts2 = time_series2.values.reshape((1, -1))
        return list(map(func, ts1, ts2))[0]


def calculate_interval(days: int) -> int:
    """If the number of days is greater than 14 (2 weeks period) then it is safer to use a bi/tri-monthly period of
    train"""
    if days == 14:
        return 2
    elif days in (21, 28):
        return 3
    else:
        return 1


def calculate_month_end(month_start: datetime, end_date: datetime, interval: int) -> datetime:
    """Calculates whether the end of the month date is either the end date or not"""
    if (month_start - end_date).days in (-29, 30):
        return end_date
    else:
        return month_start + relativedelta(months=interval) - timedelta(days=1)


def calculate_auxiliary_date(d1: datetime, days: int, month_end: datetime) -> datetime:
    """Calculates the end date of the incoming week so that it does not overstep with the end of the month date"""
    d2 = d1 + timedelta(days=days - 1)
    if d2 > month_end:
        d2 = month_end
    return d2


def calculate_d2(d1: datetime, end_date: datetime) -> datetime:
    """Calculates the end date of the second week that we are going to append into our database so that it does not
    overstep with the ending date"""
    d2 = d1 + timedelta(days=6)
    if d2 > end_date:
        d2 = end_date
    return d2


def week_frequency(days: int) -> str:
    """Based on the number of days passed it returns the corresponding frequency of weeks"""
    if days <= 7:
        return '_weekly'
    elif 14 >= days >= 7:
        return '_biweekly'
    elif 21 >= days >= 14:
        return '_triweekly'
    else:
        return '_monthly'
