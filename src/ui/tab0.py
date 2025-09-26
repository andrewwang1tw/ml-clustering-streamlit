import streamlit as st


#-----------------------------------------------------------------------------------
# Tab 0
#-----------------------------------------------------------------------------------
def tab0_show_sales_data(df_raw, df_feature):
    print("\ncorr_df, Tab 1 ===>\n", df_raw)    
    
    st.write("✏️ **原始月營收資料**")    
    st.dataframe(df_raw, height=280)
       
    st.write("🔔 **特徵**")    
    st.dataframe({
        "特徵名稱": ["continuous_flag", "avg_mom", "std_mom", "avg_yoy", "std_yoy"],
        "說明": [
            "當近 n 個月營收月增率與年增率均成長時為 1, 其餘為 0",
            "平均月增率",
            "月增率的標準差, 即波動程度",
            "平均年增率",
            "年增率的標準差, 即波動程度"
        ]
    })
        
    st.write("🔔 **加入新特徵, 以此資料作相關分析**")
    st.dataframe(df_feature, height=400)
        
