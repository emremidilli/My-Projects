# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:10 2020

@author: yunus emre midilli
"""

import pandas as pd
from Connect_to_Database import execute_sql
import numpy as np


# PREASSUMPTIONS:
    # tbl_feature_values include row data where each time stamp is correctly sequenced.

def dfGetFeatureValues(iModelId, iModelFeatureId, sFromTimeStamp = "DEFAULT", sToTimeStamp = "DEFAULT" ):
    sql = "EXEC SP_GET_TIME_STEPS "+ str(iModelId) +","+ str(iModelFeatureId)
    dfTimeSteps =  execute_sql(sql, "")
    

    if not sFromTimeStamp == "DEFAULT":
        sFromTimeStamp = "'" + str(sFromTimeStamp) + "'"
        
    if not sToTimeStamp == "DEFAULT":
        sToTimeStamp = "'" + str(sToTimeStamp) + "'"     
    
    sql_feature_value = "EXEC  [SP_GET_MODEL_FEATURE_VALUES] "+str(iModelId)+", "+ str(iModelFeatureId) +" ,"+sFromTimeStamp+", "+sToTimeStamp

    df_all_feature_values = execute_sql(sql_feature_value)
    
    df_all_feature_values = df_all_feature_values.set_index("TIME_STAMP")

    dfFeatureValues = pd.DataFrame()
    for i_index, i_row in dfTimeSteps.iterrows():
        time_step_id = i_row["ID"]
        time_step = int(i_row["TIME_STEP"])
        feature_id = i_row["FEATURE_ID"]
        boundary= i_row["BOUNDARY"]

        
        if boundary == 0:
            dfFeatureValues[time_step_id] = 0
            
        else:

            df = df_all_feature_values[df_all_feature_values["FEATURE_ID"]==feature_id]
            df_values = df["VALUE"]
            df_values = pd.DataFrame(df_values)
            
            df_values = df_values["VALUE"].shift(-time_step)
            dfFeatureValues[time_step_id] = df_values

    dfFeatureValues.sort_index(ascending=False)
    dfFeatureValues = dfFeatureValues.dropna()
    
    dfTimeSteps = dfTimeSteps.transpose()
        
    dfTimeSteps.columns = dfFeatureValues.columns 

    if iModelFeatureId == "1":
        dfFeatureValues, dfTimeSteps = dfAddSeasonalFeatures(dfFeatureValues, dfTimeSteps)

    return dfFeatureValues, dfTimeSteps



def dfAddSeasonalFeatures(dfFeatureValues, dfTimeSteps):
    dfTimeSteps = dfTimeSteps.transpose()
    aUniqueTimeSteps = dfTimeSteps.TIME_STEP.unique()
    
    exec("dfIndex = dfFeatureValues.index")
    
    
    aSeasonalFeatures = ["year","month", "day", "dayofweek", "hour"]
    
    
    iId = -1
    iModelFeatureID = -1
    
    for sSeasonalFeature in aSeasonalFeatures:
    
        for iTimeStep in aUniqueTimeSteps:

            sFeatureId = sSeasonalFeature
            
            dfTimeSteps = dfTimeSteps.append({'TIME_STEP': iTimeStep, 'MODEL_FEATURE_ID':iModelFeatureID, 'ID':iId, 'FEATURE_ID': sFeatureId}, ignore_index=True)
            
            exec("dfFeatureValues[iId] = dfIndex." + sSeasonalFeature)
            
            iId = iId -1 
            
        iModelFeatureID =  iModelFeatureID - 1 
    
            
    dfTimeSteps = dfTimeSteps.sort_values(by=['TIME_STEP', 'MODEL_FEATURE_ID'], ascending=[True, False])
    
    aColumnOrders =dfTimeSteps["ID"]
            
    dfTimeSteps = dfTimeSteps.transpose()
    
    dfFeatureValues = dfFeatureValues[aColumnOrders]

    return dfFeatureValues, dfTimeSteps
        
        

def dfGetFeatureStatistics(iFeatureID):
    sSql = "SELECT VALUE FROM TBL_FEATURE_VALUES WHERE FEATURE_ID = " + str(iFeatureID)
    dfFeatureValues = execute_sql(sSql, "")
    dfFeatureStatistics = dfFeatureValues.describe()
    dfFeatureStatistics = dfFeatureStatistics.transpose()
    return dfFeatureStatistics
    



def dfGetAllStatistics():
    sSql = "select ID, STREAM_SHORT_DESCRIPTION, SHORT_DESCRIPTION from VW_FEATURES WHERE SHORT_DESCRIPTION <> '<VOL>' order by STREAM_SHORT_DESCRIPTION, SHORT_DESCRIPTION"
    dfFeatures = execute_sql(sSql, "")
    
    dfAllStatistics = None
    for iIndex, aFeatures in dfFeatures.iterrows():

        iFeatureId = aFeatures["ID"]

        dfFeatureStatistics = dfGetFeatureStatistics(iFeatureId)
        aFeatureStatistics = dfFeatureStatistics.iloc[0]
        
        if dfAllStatistics is None:
            dfAllStatistics = pd.DataFrame(columns = dfFeatureStatistics.columns)
        
        dfAllStatistics = dfAllStatistics.append(aFeatureStatistics, ignore_index=True)
        
    dfAllStatistics[dfFeatures.columns] = dfFeatures
    

    
    return dfAllStatistics
    
    


def dfGetDimensionSize(dfTimeSteps):
    iFeatureSize = dfTimeSteps.loc[["MODEL_FEATURE_ID"]].transpose().MODEL_FEATURE_ID.unique().size
    iWindowLength = dfTimeSteps.loc[["TIME_STEP"]].transpose().TIME_STEP.unique().size
    return iFeatureSize, iWindowLength


def dfGetModels():
    sSql = "SELECT top 5 * FROM VW_MODELS order BY SHORT_DESCRIPTION desc"
    dfModels = execute_sql(sSql, "")

    return dfModels


def main(iModelId, sFromTimeStamp = "DEFAULT", sToTimeStamp = "DEFAULT" ):
    dfInput, dfTimeStepsInput= dfGetFeatureValues(iModelId, "1", sFromTimeStamp , sToTimeStamp)
    dfTarget, dfTimeStepsTarget = dfGetFeatureValues(iModelId, "2", sFromTimeStamp, sToTimeStamp)
    dfMerged =pd.merge(dfInput, dfTarget, left_index=True, right_index=True)
        
    dfInput = dfMerged[dfInput.columns]
    dfTarget= dfMerged[dfTarget.columns]
    
    return  dfInput, dfTarget ,dfTimeStepsInput, dfTimeStepsTarget
