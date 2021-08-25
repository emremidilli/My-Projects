# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 02:42:55 2020

@author: yunus
"""

import pyodbc
import pandas as pd


gc_s_CONNECTION_STRING = "Driver={SQL Server};Server=LAPTOP-LMC7DUJV\SQLEXPRESS;Database=DB_MY_PROJECT;Trusted_Connection=yes"


def execute_sql(sql_string, return_format=""):
    cnxn = pyodbc.connect(gc_s_CONNECTION_STRING)
    df = pd.DataFrame()
    
    if return_format == "":
        df = pd.read_sql(sql_string, cnxn)

    elif return_format == "none":
        cur = cnxn.cursor()
        cur.execute(sql_string)
        
    elif return_format == "json":
        df = pd.read_sql(sql_string, cnxn)
        df = df.to_json(orient='records',  date_format='iso')

    cnxn.commit()
    cnxn.close()
    return df