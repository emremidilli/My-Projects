# LIBRARIES
import pandas as pd

from datetime import datetime

import MetaTrader5 as mt5

import pytz

# CONFIGURATION
gc_o_TIME_ZONE = pytz.timezone("Etc/UTC")
gc_dt_FROM = datetime(2021, 9, 1, tzinfo=gc_o_TIME_ZONE)
gc_dt_TO = datetime(2022, 3, 10, tzinfo=gc_o_TIME_ZONE)

dfCrpytocurrencies = pd.read_csv('Data\cryptocurrencies.csv')

# FETCH FROM METATRADER 5
def aFetchFromMT5(sSymbol,dtFrom, dtTo, oFreq):
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        exit()

    aSymbolInfo = mt5.symbol_info(sSymbol)
    if not aSymbolInfo:
        print("symbol_info() failed, error code =", mt5.last_error())
        exit()

    aOhlcSample = mt5.copy_rates_range(
        sSymbol,
        oFreq,
        dtFrom, 
        dtTo
    )

    if len(aOhlcSample) == 0:
        print("copy_rates_range() failed, error code =", mt5.last_error())
        exit()

    mt5.shutdown()
    return aOhlcSample


aDatesToFetch = list(pd.date_range(start=gc_dt_FROM, end=gc_dt_TO)) #created since MT5 library fails due to time out.
aDatesSampled = aDatesToFetch[::50]
aDatesSampled.append(aDatesToFetch[-1])
aDatesSampled = list(set(aDatesSampled))
aDatesSampled.sort()

dfOhlcSource = pd.DataFrame()
for sSymbol in dfCrpytocurrencies['Symbol'].values:
    for i in range(0, len(aDatesSampled) - 1):
        dtFrom = aDatesSampled[i]
        dtTo = aDatesSampled[i+1]

        aOhlcSample = aFetchFromMT5(sSymbol,dtFrom, dtTo, oFreq = mt5.TIMEFRAME_M30)

        dfOhlcSample = pd.DataFrame(aOhlcSample)
        dfOhlcSample['symbol'] = sSymbol

        dfOhlcSample['timestamp'] = pd.to_datetime(dfOhlcSample['time'], unit= "s")
        dfOhlcSample.set_index('timestamp', inplace=True)
        dfOhlcSample.drop(["time"], axis = 1 , inplace = True)

        dfOhlcSource = dfOhlcSource.append(dfOhlcSample)

dfOhlcSource.drop_duplicates(inplace = True)

# COMPILE MARKET DATA
## Add Seasonal Features
dfOhlcSource["weekday"] = dfOhlcSource.index.weekday
dfOhlcSource["hour"] = dfOhlcSource.index.hour
dfOhlcSource["minute"] = dfOhlcSource.index.minute

## Add [Return] Feature
dfOhlcSource["return"] = (dfOhlcSource["close"] - dfOhlcSource["open"])/dfOhlcSource["open"]

## Add Candle Features
dfOhlcSource["upper_shadow"] =( dfOhlcSource["high"] - dfOhlcSource[['close', 'open']].max(axis=1))/ dfOhlcSource[['close', 'open']].max(axis=1)
dfOhlcSource["lower_shadow"] = (dfOhlcSource[['close', 'open']].min(axis=1) - dfOhlcSource["low"])/dfOhlcSource["low"]

## Transform Symbols to Columns
dfOhlc = pd.DataFrame()

i = 1
for sSymbol in dfCrpytocurrencies['Symbol'].values:
    dfSymbolValues = dfOhlcSource[dfOhlcSource['symbol'] == sSymbol]

    if i == 1:
        sHow = "right"
    else:
        sHow = "inner"
    
    dfSymbolValues = dfSymbolValues.drop('symbol', axis = 1)
    
    dfOhlc = dfOhlc.join(dfSymbolValues,how = sHow, rsuffix=sSymbol)
    
    i = i + 1
    
aColumnsOhlc = list()
for sSymbol in dfCrpytocurrencies['Symbol'].values:
    for sColumn in dfOhlcSource.columns:
        if sColumn != 'symbol':
            sNewColumn = sSymbol + ":" + sColumn
            aColumnsOhlc.append(sNewColumn)
    
dfOhlc.columns = aColumnsOhlc
dfOhlc.to_csv('Data\dfOhlc.csv')