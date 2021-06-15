"""
Neural Network classes of tensorflow.keras that work either as rescalers that transform the trending scale Google Trends
data, or as AutoEncoders that remove noise from the raw Google Trends data that are later fed into the rescaler Neural
Network models.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


class LSTM:
    """
    tf.keras Neural Network models that we can train and retrain again later on whenever there's a need for that. It
    holds the L1 and L2 Neural Networks that rescale the data for the incoming data each week into an already scaled
    database.

    Example
    -------
    >>> from datetime import datetime
    >>> import assistant_functions as af
    >>> from google_trends import Query
    >>> ts = af.retrieve_raw()  # a simple dataframe
    >>> d = ts.loc[datetime(2018, 1, 1):datetime(2018, 1, 31)]  # creating sampling data
    >>> a = ts.loc[datetime(2018, 2, 1):datetime(2018, 2, 7)]  # creating sampling data
    >>> b = Query.fetch_dataframes('ξενοδοχεια θεσσαλονικη', '2018-01-25 2018-02-07')  # creating sampling data
    >>> j = ts.loc[datetime(2018, 2, 8):datetime(2018, 2, 14)]  # incoming data
    >>> rnn = LSTM()  # initialize a 2 NN LSTM for rescale of Google Trends (with default parameters)
    >>> rnn.sampling(a, b, d)  # initialize the data (shapes and min-max scaling) to fit the model
    >>> rnn.build()  # initialize and define training methods and loss functions for each NN LSTM
    >>> rnn.train()  # fit each NN LSTM
    >>> t = rnn(j)  # rescale incoming raw Google Trends data to fit to a database
    """

    def __init__(self, layers=0, units=3, seed=None, loss='mse', optimizer='adam', flag=False, epochs=15):
        self.layers = layers
        self.units = units
        self.seed = seed
        self.X_1 = self.y_1 = self.X_2 = self.y_2 = None
        self.l1 = self.l2 = None
        self.loss = loss
        self.optimizer = optimizer
        self.flag = flag
        self.epochs = epochs

    def sampling(self, x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame):
        """We create data in order to be able to train our rescale models. We can resample later on and retrain our
        models for better data.

        Parameters
        ----------
        x:
            DataFrame - incoming isolated week data
        y:
            DataFrame - auxiliary week/s data
        z:
            DataFrame - database data
        """
        X_1, y_1, X_2, y_2 = partition_dataframes(x, y, z)
        # due to the end date sometimes the x dataframe will be
        if X_1.shape[0] != y_2.shape[0]:
            y_2 = y_2[:X_1.shape[0]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_1, y_1, X_2, y_2 = scaler.fit_transform(X_1), scaler.fit_transform(y_1), scaler.fit_transform(X_2), \
            scaler.fit_transform(y_2)
        self.X_1 = X_1.reshape(X_1.shape[0], 1, X_1.shape[1])
        self.y_1 = y_1.reshape(y_1.shape[0], 1, y_1.shape[1])
        self.X_2 = X_2.reshape(X_2.shape[0], 1, X_2.shape[1])
        self.y_2 = y_2.reshape(y_2.shape[0], 1, y_2.shape[1])

    def build(self):
        """Defines the layers for the rescaling method"""
        self.l1_()
        self.l2_()

    def l1_(self):
        """Defines the Neural Network for the first layer of rescaling"""
        self.l1 = self._build(self.y_1)

    def l2_(self):
        """Defines the Neural Network for the second layer of rescaling"""
        self.l2 = self._build(self.y_2)

    def _build(self, labels: np.array):
        """Creates a Recurrent Neural Network"""
        if self.seed is not None:
            tf.random.set_seed(self.seed)
        model = tf.keras.Sequential([tf.keras.layers.LSTM(units=self.units,
                                                          return_sequences=True,
                                                          input_shape=(labels.shape[1], labels.shape[2]))])
        for i in range(self.layers):
            model.add(tf.keras.layers.LSTM(units=self.units, return_sequences=True))
        model.add(tf.keras.layers.Dense(units=1))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def train(self):
        """Trains the L1 and L2 models for our rescale pipeline"""
        try:
            if self.flag:
                _ = self.l1.fit(self.X_1, self.y_1, epochs=self.epochs, verbose=0, shuffle=False)
                y = self.l1.predict(self.X_1)
                _ = self.l2.fit(y.reshape(y.shape[0], 1, y.shape[1]), self.y_2, epochs=self.epochs, verbose=0,
                                shuffle=False)
            else:
                _ = self.l1.fit(self.X_1, self.y_1, epochs=self.epochs, verbose=0, shuffle=False)
                _ = self.l2.fit(self.X_2, self.y_2, epochs=self.epochs, verbose=0, shuffle=False)
        except AttributeError:
            print('Models have not be trained yet. Run sampling, build and then train methods.')

    def __call__(self, isolated_table: pd.DataFrame):
        """Rescales and returns the isolated_table, which contains the incoming week's data, to the scale of our
        database.

        Parameters
        ----------
        isolated_table:
            DataFrame - incoming week to be rescaled
        Returns
        -------
        isolated_table:
            DataFrame - contains the data that has been rescaled to fit the database
        """
        X_ = isolated_table.copy()
        X_ = X_.values.reshape((X_.shape[0], 1, X_.shape[1])).astype(np.float32)
        try:
            y = self.l1.predict(X_)
        except AttributeError:
            print('Models have not be trained yet. Run sampling, build, train and then rescale methods.')
            return

        predictions = self.l2.predict(y.reshape(y.shape[0], 1, y.shape[1]))
        scaler = MinMaxScaler(feature_range=(0, 100))
        predictions = predictions.reshape(y.shape[0], y.shape[1])
        predictions = scaler.fit_transform(predictions)

        return pd.DataFrame(data=predictions, index=isolated_table.index, columns=[isolated_table.keys()[0]])

    def __str__(self):
        return 'LSTM {\n\tNumber of intermediate layers: %s, \n\tNumber of cells: %s\n\tLoss function: %s' \
               '\n\tOptimizer:%s\n}' % (str(self.layers), str(self.units), self.loss, self.optimizer)


class AutoEncoder:
    """
    LSTM AutoEncoder model that rescales raw Google Trends data

    Example
    -------
    >>> from datetime import datetime
    >>> import assistant_functions as af
    >>> from google_trends import Query
    >>> ts = af.retrieve_raw()  # a simple dataframe
    >>> d = ts.loc[datetime(2018, 1, 1):datetime(2018, 1, 31)]  # creating sampling data
    >>> a = ts.loc[datetime(2018, 2, 1):datetime(2018, 2, 7)]  # creating sampling data
    >>> b = Query.fetch_dataframes('ξενοδοχεια θεσσαλονικη', '2018-01-25 2018-02-07')  # creating sampling data
    >>> j = ts.loc[datetime(2018, 2, 8):datetime(2018, 2, 14)]  # incoming data
    >>> ae = AutoEncoder()  # initialize a 2 NN LSTM for rescale of Google Trends (with default parameters)
    >>> ae.sampling(a, b, d)  # initialize the data (shapes and min-max scaling) to fit the model
    >>> ae.build()  # initialize and define training methods and loss functions for each NN LSTM
    >>> ae.train()  # fit each NN LSTM
    >>> t = ae(j)  # rescale incoming raw Google Trends data to fit to a database
    """

    def __init__(self, units=32, seed=None, loss='mae', optimizer='adam', flag=True, epochs=30):
        self.units = units
        self.seed = seed
        self.X_1 = self.y_1 = self.X_2 = self.y_2 = None
        self.l1 = self.l2 = None
        self.loss = loss
        self.optimizer = optimizer
        self.flag = flag
        self.epochs = epochs

    def sampling(self, x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame):
        """We create data in order to be able to train our rescale models. We can resample later on and retrain our
        models for better data.

        Parameters
        ----------
        x:
            DataFrame - incoming isolated week data
        y:
            DataFrame - auxiliary week/s data
        z:
            DataFrame - database data
        """
        X_1, y_1, X_2, y_2 = partition_dataframes(x, y, z)

        # due to the end date sometimes the x dataframe will be
        if X_1.shape[0] != y_2.shape[0]:
            y_2 = y_2[:X_1.shape[0]]
        print(len(x), len(y), len(z), X_1.shape, y_1.shape, X_2.shape, y_2.shape)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_1, y_1, X_2, y_2 = scaler.fit_transform(X_1), scaler.fit_transform(y_1), scaler.fit_transform(X_2), \
            scaler.fit_transform(y_2)
        self.X_1 = X_1.reshape(X_1.shape[0], 1, X_1.shape[1])
        self.y_1 = y_1.reshape(y_1.shape[0], 1, y_1.shape[1])
        self.X_2 = X_2.reshape(X_2.shape[0], 1, X_2.shape[1])
        self.y_2 = y_2.reshape(y_2.shape[0], 1, y_2.shape[1])

    def build(self):
        """Defines the layers for the rescaling method"""
        self.l1_()
        self.l2_()

    def l1_(self):
        """Defines the Network for the first layer of rescaling"""
        self.l1 = self._build(self.y_1)

    def l2_(self):
        """Defines the Network for the second layer of rescaling"""
        self.l2 = self._build(self.y_2)

    def _build(self, labels: np.array):
        """Creates an LSTM AutoEncoder Network"""
        if self.seed is not None:
            tf.random.set_seed(self.seed)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(self.units, return_sequences=True, input_shape=(labels.shape[1], labels.shape[2])),
            tf.keras.layers.LSTM(self.units // 2, return_sequences=False,
                                 input_shape=(labels.shape[1], labels.shape[1])),
            tf.keras.layers.RepeatVector(labels.shape[1]),
            tf.keras.layers.LSTM(self.units // 2, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(self.units, activation='relu', return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(labels.shape[2]))
        ])
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def train(self):
        """Trains the L1 and L2 models for our rescale pipeline"""
        try:
            if self.flag:
                _ = self.l1.fit(self.X_1, self.y_1, epochs=self.epochs, shuffle=False)
                y = self.l1.predict(self.X_1)
                _ = self.l2.fit(y.reshape(y.shape[0], 1, y.shape[1]), self.y_2, epochs=self.epochs, shuffle=False)
            else:
                _ = self.l1.fit(self.X_1, self.y_1, epochs=self.epochs, verbose=0, shuffle=False)
                _ = self.l2.fit(self.X_2, self.y_2, epochs=self.epochs, verbose=0, shuffle=False)
        except AttributeError:
            print('Models have not be trained yet. Run sampling, build and then train methods.')

    def __call__(self, isolated_table: pd.DataFrame):
        """Rescales and returns the isolated_table, which contains the incoming week's data, to the scale of our
        database.

        Parameters
        ----------
        isolated_table:
            DataFrame - incoming week to be rescaled
        Returns
        -------
        isolated_table:
            DataFrame - contains the data that has been rescaled to fit the database
        """
        X_ = isolated_table.copy()
        X_ = X_.values.reshape((X_.shape[0], 1, X_.shape[1])).astype(np.float32)
        try:
            y = self.l1.predict(X_)
        except AttributeError:
            print('Models have not be trained yet. Run sampling, build, train and then rescale methods.')
            return

        predictions = self.l2.predict(y.reshape(y.shape[0], 1, y.shape[1]))
        scaler = MinMaxScaler(feature_range=(0, 100))
        predictions = predictions.reshape(y.shape[0], y.shape[1])
        predictions = scaler.fit_transform(predictions)

        return pd.DataFrame(data=predictions, index=isolated_table.index, columns=[isolated_table.keys()[0]])


class LstmAE(tf.keras.Model):
    """
    An LSTM AutoEncoder for denoising data.
    """

    def __init__(self, *shape, neurons=128):
        super(LstmAE, self).__init__()
        self.il = tf.keras.layers.InputLayer(input_shape=(shape[0], shape[1]))
        self.lstm1 = tf.keras.layers.LSTM(neurons, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(neurons // 2, return_sequences=False)
        self.rp = tf.keras.layers.RepeatVector(shape[0])
        self.lstm3 = tf.keras.layers.LSTM(neurons // 2, return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(neurons, return_sequences=True)
        self.td = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(shape[1]))

    def call(self, inputs, training=None, mask=None):
        """Transforms the input into the shape of each layer sequentially"""
        x = self.il(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.rp(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        return self.td(x)


def partition_dataframes(isolated_df: pd.DataFrame, auxiliary_df: pd.DataFrame, database: pd.DataFrame):
    """Partitions and returns the corresponding values of isolated_df and isolated_df inside the auxiliary_df, and
    auxiliary_df and auxiliary_df corresponding to the database.

    Parameters
    ----------
    isolated_df:
        DataFrame - contains the values of the incoming week we want to update
    auxiliary_df:
        DataFrame - contains the values of the incoming week plus 1 week prior to that
    database:
        DataFrame - contains the database up until a day before the incoming week starts. Also contains the 1 week
        that the auxiliary has in it.

    Returns
    -------
    X_1, y_1, X_2, y_2:
        nparray - contains values that correspond to isolated_df, auxiliary_df and database
    """
    X_1 = isolated_df.loc[isolated_df.index[0]:auxiliary_df.index[-1]].values
    y_1 = auxiliary_df.loc[isolated_df.index[0]:auxiliary_df.index[-1]].values
    X_2 = auxiliary_df.loc[auxiliary_df.index[0]:database.index[-1]].values
    y_2 = database.loc[auxiliary_df.index[0]:database.index[-1]].values
    return X_1, y_1, X_2, y_2
