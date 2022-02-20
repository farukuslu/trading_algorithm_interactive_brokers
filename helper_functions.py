# Contains helper functions for your apps!
from os import listdir, remove

# If the io following files are in the current directory, remove them!
# 1. 'currency_pair.txt'
# 2. 'currency_pair_history.csv'
# 3. 'trade_order.p'
def check_for_and_del_io_files():
    # Your code goes here.
    dir = listdir()
    file1, file2, file3 = 'training_data.txt', 'ticker_n.txt', 'data.csv'
    if file1 in dir:
        remove(file1)
    if file2 in dir:
        remove(file2)
    if file3 in dir:
        remove(file3)
    pass # nothing gets returned by this function, so end it with 'pass'.