import pandas as pd
import os
import Connect_to_Database


gc_s_FOLDER_PATH = r'C:\Users\yunus\Documents\2020-2021 RTU\Second Semester\Evolutionary and Genetic Algorithms\Data'


def sGetStreamID(sStream):
    sSqlSelect = "SELECT * FROM TBL_STREAMS WHERE SHORT_DESCRIPTION = '"+ sStream  +"'"
    
    dfStreamInDb = Connect_to_Database.execute_sql(sSqlSelect)
    
    if len(dfStreamInDb) == 0:
        sSqlInsert = "INSERT INTO TBL_STREAMS (SHORT_DESCRIPTION) VALUES ('"+ sStream  +"')"
        
        Connect_to_Database.execute_sql(sSqlInsert, "none")
        
        dfStreamInDb = Connect_to_Database.execute_sql(sSqlSelect)
     
    sStreamID = str(dfStreamInDb["ID"][0])
    
    return sStreamID



def sGetFeatureID(sFeature, sStreamID):
    sSqlSelect = "SELECT * FROM TBL_FEATURES WHERE SHORT_DESCRIPTION = '"+ sFeature  +"' AND STREAM_ID = " + str(sStreamID)
    
    dfFeatureInDb = Connect_to_Database.execute_sql(sSqlSelect)
    
    if len(dfFeatureInDb) == 0:
        sSqlInsert = "INSERT INTO TBL_FEATURES (SHORT_DESCRIPTION, STREAM_ID) VALUES ('"+ sFeature  +"', " + str(sStreamID) +")"
        
        Connect_to_Database.execute_sql(sSqlInsert, "none")
        
        dfFeatureInDb = Connect_to_Database.execute_sql(sSqlSelect)
     
    sFeatureID = str(dfFeatureInDb["ID"][0])
    
    return sFeatureID    


def AddRecordToDb(sFeatureID, sTimeStamp, decFeatureValue):
    
    sSql = "INSERT INTO TBL_FEATURE_VALUES (TIME_STAMP, FEATURE_ID, VALUE) VALUES ('"+ sTimeStamp  +"', "+ sFeatureID +", "+ str(decFeatureValue) +")"
    Connect_to_Database.execute_sql(sSql, "none")
    
    
        
def main():

    aFileNames = os.listdir(gc_s_FOLDER_PATH)
    
    for sFileName in aFileNames:
        sFilePath = gc_s_FOLDER_PATH + r'/' + sFileName
        
        sStream = sFileName[0:6]
        
        sStreamID = sGetStreamID(sStream)
        
        dfRowData = pd.read_csv(sFilePath, delimiter = '\t')
        
        dfRowData["TIMESTAMP"] = dfRowData["<DATE>"] + ' '  +dfRowData["<TIME>"]
        
        dfRowData = dfRowData.drop(["<DATE>", "<TIME>"], axis=1)
        
        dfRowData =  dfRowData.set_index("TIMESTAMP")
        
        aFeatures = dfRowData.columns
                
        for sFeature in aFeatures:
            
            sFeatureID = sGetFeatureID(sFeature, sStreamID)
            
            for iIndex, aRow in dfRowData.iterrows():
                
                sTimeStamp = iIndex
                
                decFeatureValue = aRow[sFeature]
                
                AddRecordToDb(sFeatureID, sTimeStamp, decFeatureValue)
                
                
main()