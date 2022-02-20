# SimpleHistoryExample.py
from hw2_utils import *


def main():
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

        bbg_identifier = "IVV US Equity"

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
        request.set("startDate", "20210326")
        request.set("endDate", "20210330")
        request.set("maxDataPoints", 1400) # Don't adjust please :)

        print("Sending Request:", request)
        # Send the request
        session.sendRequest(request)

        # Process received events
        while(True):
            # We provide timeout to give the chance for Ctrl+C handling:
            ev = session.nextEvent(500)
            for msg in ev:
                if str(msg.messageType()) == "HistoricalDataResponse":

                    histdata = []

                    for fd in msg.getElement("securityData").getElement(
                        "fieldData").values():
                        histdata.append([fd.getElementAsString("date"), \
                        fd.getElementAsFloat("OPEN"), fd.getElementAsFloat(
                            "HIGH"), fd.getElementAsFloat("LOW"), \
                        fd.getElementAsFloat("PX_LAST"), \
                        fd.getElementAsFloat("EQY_WEIGHTED_AVG_PX")])

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

if __name__ == "__main__":
    print("SimpleHistoryExample")
    try:
        hd = main()
        print(hd)
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Stopping...")
