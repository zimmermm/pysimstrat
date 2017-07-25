import numpy as np
from datetime import datetime


def datetime64_to_days(datetime64):
    return datetime64.astype(np.int64)/1e9/3600/24.


def days_to_datetime64(days):
    return np.datetime64(datetime.utcfromtimestamp(days*24*3600), 'ns')


def days_from_1970_to_days_from_year_one(days):
    return (datetime.fromtimestamp(days*24*3600)-datetime(1,1,1)).total_seconds()/3600/24.
