# For utilities (helper functions) to be used in this HW.

# For Bloomberg ----------------------------------------------------------------
########## Comment this section out if on Mac or run in a Windows VM. ##########
# taken from SimpleHistoryExample.py
from __future__ import print_function
from __future__ import absolute_import

from optparse import OptionParser

import os
import platform as plat
import sys
if sys.version_info >= (3, 8) and plat.system().lower() == "windows":
    # pylint: disable=no-member
    with os.add_dll_directory(os.getenv('BLPAPI_LIBDIR')):
        import blpapi
else:
    import blpapi
import re

def parseCmdLine():
    parser = OptionParser(description="Retrieve reference data.")
    parser.add_option("-a",
                      "--ip",
                      dest="host",
                      help="server name or IP (default: %default)",
                      metavar="ipAddress",
                      default="localhost")
    parser.add_option("-p",
                      dest="port",
                      type="int",
                      help="server port (default: %default)",
                      metavar="tcpPort",
                      default=8194)

    (options, args) = parser.parse_args()

    return options

def req_historical_data(bbg_identifier, startDate, endDate):
    options = parseCmdLine()

    # Fill SessionOptions
    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost(options.host)
    sessionOptions.setServerPort(options.port)

    print("Connecting to %s:%s" % (options.host, options.port))
    # Create a Session
    session = blpapi.Session(sessionOptions)

    # Start a Session
    if not session.start():
        print("Failed to start session.")
        return

    try:
        # Open service to get historical data from
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        # Obtain previously opened service
        refDataService = session.getService("//blp/refdata")

        # Create and fill the request for the historical data
        request = refDataService.createRequest("HistoricalDataRequest")
        request.getElement("securities").appendValue(bbg_identifier)
        request.getElement("fields").appendValue("OPEN")
        request.getElement("fields").appendValue("HIGH")
        request.getElement("fields").appendValue("LOW")
        request.getElement("fields").appendValue("PX_LAST")
        request.getElement("fields").appendValue("EQY_WEIGHTED_AVG_PX")
        request.set("periodicityAdjustment", "ACTUAL")
        request.set("periodicitySelection", "DAILY")
        request.set("startDate", re.sub("-", "", startDate))
        request.set("endDate", re.sub("-", "", endDate))
        request.set("maxDataPoints", 1400) # Don't adjust please :)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)

        # Process received events
        while (True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":

                    histdata = []

                    for fd in msg.getElement("securityData").getElement(
                            "fieldData").values():
                        histdata.append([fd.getElementAsString("date"), \
                                         fd.getElementAsFloat("OPEN"),
                                         fd.getElementAsFloat(
                                             "HIGH"),
                                         fd.getElementAsFloat("LOW"), \
                                         fd.getElementAsFloat("PX_LAST"), \
                                         fd.getElementAsFloat(
                                             "EQY_WEIGHTED_AVG_PX")])

                    histdata = pd.DataFrame(histdata, columns=["Date",
                                                               "Open",
                                                               "High", "Low",
                                                               "Close", "VWAP"])

            if ev.eventType() == blpapi.Event.RESPONSE:
                # Response completely received, so we could exit
                return histdata
    finally:
        # Stop the session
        session.stop()

__copyright__ = """
Copyright 2012. Bloomberg Finance L.P.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:  The above
copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

####### End of Bloomberg Section -----------------------------------------------
################################################################################

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

def to_years(x):
    str_split = x.lower().split()
    if len(str_split) == 2:
        if str_split[1] == 'mo':
            return int(str_split[0]) / 12
        if str_split[1] == 'yr':
            return int(str_split[0])

def fetch_usdt_rates(YYYY):
    # Requests the USDT's daily yield data for a given year. Results are
    #   returned as a DataFrame object with the 'Date' column formatted as a
    #   pandas datetime type.

    URL = 'https://www.treasury.gov/resource-center/data-chart-center/' + \
          'interest-rates/pages/TextView.aspx?data=yieldYear&year=' + str(YYYY)

    cmt_rates_page = requests.get(URL)

    soup = BeautifulSoup(cmt_rates_page.content, 'html.parser')

    table_html = soup.findAll('table', {'class': 't-chart'})

    df = pd.read_html(str(table_html))[0]
    df.Date = pd.to_datetime(df.Date)

    return df

def Y_m_d_to_unix_str(ymd_str):
    return str(int(time.mktime(pd.to_datetime(ymd_str).date().timetuple())))

def fetch_GSPC_data(start_date, end_date):
    # Requests the USDT's daily yield data for a given year. Results are
    #   returned as a DataFrame object with the 'Date' column formatted as a
    #   pandas datetime type.

    URL = 'https://finance.yahoo.com/quote/%5EGSPC/history?' + \
            'period1=' + Y_m_d_to_unix_str(start_date) + \
            '&period2=' + Y_m_d_to_unix_str(end_date) + \
            '&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'

    gspc_page = requests.get(URL)

    soup = BeautifulSoup(gspc_page.content, 'html.parser')

    table_html = soup.findAll('table', {'data-test': 'historical-prices'})


    df = pd.read_html(str(table_html))[0]

    df.drop(df.tail(1).index, inplace=True)

    # see formats here: https://www.w3schools.com/python/python_datetime.asp
    df.Date = pd.to_datetime(df.Date)

    return df
