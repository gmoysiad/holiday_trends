"""
Python module that contains functions that help other major classes
"""

import hashlib
import itertools as it
import json
import os
import time
from datetime import datetime, timedelta
from dateutil.rrule import rrule, WEEKLY

import pandas as pd
import sqlalchemy as db
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq

import google_trends as gt


def create_connection():
    """An auxialiry function that connects to the database.

    Using enviromental variables, that have stored the URL for the connection, the username, password, host URL, port
    number and the specified database that we need to connect, we establish the connection with MySQL database.

    Returns
    -------
    tuple:
        Engine - establishes the connection with the database.
        Connection - enables the communication between the user and the database.
        Metadata - holds the tables and schemas along with other valuable information of the database.
    """
    url = os.getenv('DB_URL') % (os.getenv('DB_USER'), os.getenv('DB_PASSWORD'), os.getenv('DB_HOST'),
                                 os.getenv('DB_PORT'), os.getenv('DB_DATABASE'))
    engine = db.create_engine(url)
    connection = engine.connect()
    metadata = db.MetaData()
    metadata.reflect(engine)  # make tables and schemas visible
    return engine, connection, metadata


def close_connection(engine, connection):
    """An auxialiry function that closes the connection to the database"""
    connection.close()
    engine.dispose()


def retrieve_scaled(hashed_query, destination):
    """Retrieves the scaled Google Trends data from the database, if they don't exist shows a message"""
    engine, connection, metadata = create_connection()

    query = db.text("SELECT data FROM aihg_scaled_trends WHERE hash = :x AND destination = :y")
    data = list(connection.execute(query, x=hashed_query, y=destination))

    connection.close()
    engine.dispose()

    if data:
        return string_to_dataframe(data[0][0])
    else:
        print('Scaled Google Trends do not exist. Creating scaled data method is initialized.')
        return None


def retrieve_raw(hashed_query, destination):
    """Retrieves the raw Google Trends data from the database"""
    engine, connection, _ = create_connection()

    query = db.text("SELECT data FROM aihg_raw_trends WHERE hash = :x AND destination = :y AND frequency = 7")
    data = list(connection.execute(query, x=hashed_query, y=destination))[0]

    connection.close()
    engine.dispose()

    return string_to_dataframe(data[0])


def string_to_dataframe(data: str):
    """Converts string to JSON object and then to pandas DataFrame"""
    temp_json = json.loads(data)
    return pd.read_json(temp_json)


def create_hash(query: str):
    """Creates and returns a has form for a string input
    Returns
    -------
    hash:
        string - hash of the query in string form
    """
    return hashlib.md5(query.encode()).hexdigest()


def transform_dataframe(df: pd.DataFrame, frequency) -> pd.DataFrame:
    """Given a dataframe and a frequency that the data was captured returns a transposed dataframe with additional
    indexes indicating also the frequency that the data was captured."""
    frequency = gt.week_frequency(frequency)
    frequency = [[frequency] * len(df)][0]
    end_indexes = list(map(aux_timestamps, df.index))
    tuple_indexes = list(it.zip_longest(frequency, df.index, end_indexes))
    indexes = pd.MultiIndex.from_tuples(tuple_indexes, names=['Frequency', 'Start', 'End'])
    df = df.set_index(indexes)
    return df.transpose()


def aux_timestamps(start_index: datetime) -> datetime:
    return start_index + timedelta(seconds=59, minutes=59, hours=23)


def fetch_dataframes(query: str, timeframe: str, location: str = 'GR'):
    """Returns from Google Trends a dataframe containing the interest_over_time of a query during a specific timeframe
    based on a geolocation. If, during that timeframe, it returns an empty dataframe, because there was no data to be
    found, we create our own dataframe containing zeros for that timeframe.

    Parameters
    ----------
    query:
        string - query-search_term that will base the data collection
    timeframe:
        string - we pass a timeframe that has been operated on with datetime and then we convert it to string
    location:
        string - the location we want to base our search, e.g. a country or a worldwide base search

    Returns
    -------
    int_over_time:
        DataFrame - contains the interest_over_time (popularity) of a query for the specific timeframe
    """
    pytrends = TrendReq()
    print(query, timeframe, location)
    try:
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=location)
        time.sleep(3)
    except ResponseError:
        print('Too many requests to Google Trends. Google Trends disconnected.')
    except ConnectionError:
        print('Connection timed out, please wait...')
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=location)

    # if a week has 0 interest_over_time pytrends will return an empty dataframe and it's raising KeyError because
    # it's an empty dataframe there is no column=isPartial for it to drop
    try:
        int_over_time = pytrends.interest_over_time()
        int_over_time.drop('isPartial', axis=1, inplace=True)
    except KeyError:
        # create an index based on the timeframe that we have passed above
        str_idx = datetime.strptime(timeframe.split(' ')[0], '%Y-%m-%d')
        end_idx = datetime.strptime(timeframe.split(' ')[1], '%Y-%m-%d')
        index = pd.date_range(str_idx, end_idx, freq='D')

        # create and return a dataframe with 0s based on the index that we created above
        int_over_time = pd.DataFrame(index=index, columns=[query]).fillna(0)
    return int_over_time


def create_raw_database(hashed_query, query, start, end, origin, language_code):
    """Creates a database full of raw Google Trends data based on starting and ending dates"""
    print('Fetching raw Google Trends data for:', query, hashed_query, language_code, origin)

    # Create a list with weekly dates '1/1/18 7/1/18', '8/1/18 14/1/18' etc.
    weekly_dates = list(rrule(freq=WEEKLY, dtstart=start, until=end))

    # Iterate through the list of weekly_dates and create a list with 1/2/3/4-week dates
    date_combos = []
    for frequency in range(1, 5):
        for dt in weekly_dates[::frequency]:
            date_combos.append(
                dt.strftime('%Y-%m-%d') + ' ' + (dt + timedelta(days=(frequency * 7) - 1)).strftime('%Y-%m-%d'))

    # Creates combinations of the queries with each date from date_combos
    # (query, geolocation, date)
    all_search_terms = list(it.product([query], date_combos, [origin]))
    dataframes = list(map(lambda x: (fetch_dataframes(x[0], x[1], x[2]), x[2]), all_search_terms))

    # Creates a dictionary with keys the number of days in the fetch
    # dict = {'7-geolocation': dataframe, '14-geolocation': dataframe, '21-geolocation': dataframe,
    #         '28-geolocation': dataframe
    all_df = {str(i[0]) + '-' + i[1]: pd.DataFrame() for i in
              set(list(map(lambda x: (len(x[0]), x[1]), dataframes)))}
    for df, geolocation in dataframes:
        all_df[str(len(df)) + '-' + geolocation] = all_df[str(len(df)) + '-' + geolocation].append(df)

    engine, connection, metadata = create_connection()
    for key, value in all_df.items():
        week_days, geolocation = int(key.split('-')[0]), key.split('-')[1]
        if geolocation == '':
            geolocation = 'Worldwide'

        df_json = value.to_json()
        insert = db.insert(metadata.tables['aihg_raw_trends']).values(
            hash=hashed_query, frequency=week_days, start=start, end=end, origin=geolocation,
            language_code=language_code, destination=query.split()[1], data=df_json
        )
        _ = connection.execute(insert)
    close_connection(engine, connection)

    return all_df
