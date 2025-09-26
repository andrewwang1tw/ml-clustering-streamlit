import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import os

#--------------------------------------------------------------------------------
# make_engine
#--------------------------------------------------------------------------------
@st.cache_resource
def make_engine(server, database, driver='ODBC Driver 17 for SQL Server', trusted=True, user=None, password=None):
    if trusted:
        conn_str = f"mssql+pyodbc://{server}/{database}?trusted_connection=yes&driver={driver}"
    else:
        conn_str = f"mssql+pyodbc://{user}:{password}@{server}/{database}?driver={driver}"
    return create_engine(conn_str)

#--------------------------------------------------------------------------------
# load_sales_data
#--------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_sales_data(_engine, year=114, month_from=1):
    query = f"""
    SELECT 公司代號 AS stock_id, 公司名稱 AS stock_name, (年+1911)*100+月 as YM,
        上月比較增減百分比 as MoM,
        去年同月增減百分比 as YoY,
        當月營收 SALE
    FROM T_SALES_M
    WHERE 年 = {year} AND 月 >= {month_from} AND LEN(公司代號)=4
    ORDER BY 公司代號, 年, 月 DESC
    """
    df = pd.read_sql(query, _engine)
    return df


def load_sales_data_from_csv():
    #file_path = "data/stock.csv"    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    file_path = os.path.join(script_dir, "stock.csv")
    try:
        df = pd.read_csv(file_path)     
        print("\n已成功讀取 Excel 檔案：", file_path)
        return df
    
    except FileNotFoundError:
        print(f"\n錯誤：找不到檔案 '{file_path}'。請確認檔案已上傳至 GitHub 儲存庫。")
    except Exception as e:
        print(f"\n發生錯誤：{e}")
    
    return None