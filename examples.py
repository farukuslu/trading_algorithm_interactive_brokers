# Imports

# why don't we need to import pandas as pd?
# because hw2_utils does that!
from pandas import util
from hw2_utils import *
import time
from datetime import datetime, date

# Make two data frames for testing
df1 = util.testing.makeDataFrame().head()
df2 = util.testing.makeDataFrame().head()

# Print dataframes
print(df1)
print(df2)

# Print row names only
# here we coerce the index to a list
print(list(df1.index))
print(list(df2.index))

# Print column names only
# here we coerce the index to a list
print(list(df1.columns))
print(list(df2.columns))

# Bind data frames by row:
df_bound_by_row = pd.concat([df1, df2], axis = 0)
print(df_bound_by_row)

# Bind columns:
df_bound_by_col = pd.concat([df1, df2], axis = 1)
print(df_bound_by_col)

# Doesn't bind cleanly because different row names.
# Set the rownames of df2 to those of df1
df2.index = df1.index
print(list(df2.index))
# now try again...
df_bound_by_col = pd.concat([df1, df2], axis = 1)
print(df_bound_by_col)
# ..but now we've got columns with the same name!
# so, better to keep track of columns:
df1 = df1.add_suffix('_1')
df2 = df2.add_suffix('_2')
# now try:
df_bound_by_col = pd.concat([df1, df2], axis = 1)
print(df_bound_by_col)

#### Doing things to data frames ###############################################

# First, let's pull some info from the USDT site
cmt_rates_2021 = fetch_usdt_rates(2021)
time.sleep(5) # sleep 5 sec to give the USDT website a break
cmt_rates_2020 = fetch_usdt_rates(2020)
time.sleep(5)
cmt_rates_2019 = fetch_usdt_rates(2019)

# bind by rows and sort by date!
cmt_rates = pd.concat([cmt_rates_2019, cmt_rates_2021, cmt_rates_2020], axis=0)
print(cmt_rates.tail()) # cmt_rates isn't sorted by Date...
cmt_rates = cmt_rates.sort_values('Date') # so sort it!
print(cmt_rates.head()) # looking good.

# convert a date to a Unix timestamp
# Unix timestamp: number of seconds from 01 Jan 1970 00:00:00 to 'now'.
print(date.today()) # today's date
print(time.mktime(date.today().timetuple())) # today's date as a UNIX timestamp

# convert a Unix timestamp to a date
print(datetime.fromtimestamp(1616457600).date()) # what does the '.date()' do?

# convert a date STRING to a datetime:
print(pd.to_datetime("2021-03-21").date())




