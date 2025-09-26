import streamlit as st
from ml.clustering import run_analysis

#import ui_elements as ui
import ui.ui_main as ui
from ui.tab0 import tab0_show_sales_data
from ui.tab1 import tab1_show_correlation_heatmap
from ui.tab2 import tab2_show_elbow_plot, tab2_show_elbow_plot2
from ui.tab3 import tab3_show_pca_2d, tab3_show_pca_3d
from ui.tab4 import tab4_show_cluster_feature_boxplots, tab4_show_cluster_summary
from ui.tab5 import tab5_show_cluster_details
from ui.tab6 import tab6_show_cluster_mom_yoy

def main():
    ui.set_font()
    ui.set_layout()

    # --- 取得 sidebar 參數, 當 user 按 run 的時候執行 ---
    params = ui.sidebar_inputs()    
    if params['run']:
        #with st.spinner('連線、擷取資料中...'):               
        run_analysis(params)            
        #st.success('分析完成，請下拉選群集來查看細節。')

    # --- 若 session_state 已經有分析結果，顯示結果 ---
    if 'df_clustered' in st.session_state:
        df_clustered = st.session_state['df_clustered']
        inertias = st.session_state['inertias']
        use_k = st.session_state['use_k']
        elbow_k = st.session_state['elbow_k']
        corr_before = st.session_state['corr_before']
        corr_after = st.session_state['corr_after']
        features = st.session_state['features']
        dropped = st.session_state['dropped']
        params_saved = st.session_state['params']
        df_raw = st.session_state['df_raw']
        df_feature = st.session_state['df_feature']
        
        # 建立 tabs
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📅 資料",
            "📊 相關矩陣",
            "📉 Elbow",
            "🔵 PCA 分析",
            "📋 群集特徵",
            "🗓️ 分群",            
            "📈 月增/年增率"
        ])

        # Tab 0: 資料  
        with tab0:
            #ui.show_sales_data(df_raw, df_feature)
            tab0_show_sales_data(df_raw, df_feature)
            
        with tab1:
            #ui.show_correlation_heatmap(corr_before, corr_after, dropped=dropped)
            tab1_show_correlation_heatmap(corr_before, corr_after, dropped=dropped)
            
        # Tab2: Elbow
        with tab2:
            #ui.show_elbow_plot(params_saved['k_min'], params_saved['k_max'], inertias, use_k, elbow_k)
            tab2_show_elbow_plot2(params_saved['k_min'], params_saved['k_max'], inertias, use_k, elbow_k)

        # Tab3: PCA
        with tab3:
            #ui.show_pca_3d(df_clustered, pca3=st.session_state.get('pca3', None), features=features)
            #ui.show_pca_2d(df_clustered, pca2=st.session_state.get('pca2', None), features=features)  
            tab3_show_pca_3d(df_clustered, pca3=st.session_state.get('pca3', None), features=features)
            tab3_show_pca_2d(df_clustered, pca2=st.session_state.get('pca2', None), features=features)  
            
        # Tab4: 群集特徵 summary + boxplots
        with tab4:
            #ui.show_cluster_summary(df_clustered, features)
            #ui.show_cluster_feature_boxplots(df_clustered, params_saved['last_n_months'])
            tab4_show_cluster_summary(df_clustered, features)
            tab4_show_cluster_feature_boxplots(df_clustered, params_saved['last_n_months'])

        # Tab5: 群集詳細
        with tab5:
            #ui.show_cluster_details(df_clustered)
            tab5_show_cluster_details(df_clustered)
            
        # Tab6: MoM, YoY
        with tab6:
            tab6_show_cluster_mom_yoy(df_clustered)    

    else:        
        st.header("""使用 Python, LangGraph, Llama3.1, Streamlit 建立一個使用 Machine Learning 分群的股票營收分析工具,來幫助我們快速找出潛力股.""") 
        st.markdown( """🌟 傳統上，我們需要花很多時間手動篩選股票，而營收報表通常是靜態的，難以找出趨勢或相似的公司。
            我們將透過以下 step 將台股 1,800 多家公司依月營收資料 分成不同屬性的**群集**。
            """) 
        
        with st.expander('資料準備'):
            st.markdown("""🔑 從資料庫抓 台股(約 1,800 檔公司), 自 2025/1 月-2025/8 月的月營收資料, 並整理之。""" )
        
        with st.expander('特徵工程與相關矩陣'):
            st.markdown("""🔑 特徵   
                **continuous_flag** : 最近 n 個月, 月營收的月增率與年增率均成長時則為 1 。   
                **avg_mom** : 平均月增率, 反映公司成長動能。   
                **std_mom** : 月增率的標準差, 反映公司營收的波動性。   
                **avg_yoy** : 平均年增率, 反映公司成長動能。      
                **std_yoy** : 年增率的標準差, 反映公司營收的波動性。                  
                """)            
            st.markdown("""🔑 然後作相關分析, 移除高度相關的**特徵**""")
            
        with st.expander('Elbow 圖： 決定最佳分群數 k'):
            st.markdown('🔑 透過 Elbow method 決定最佳分群數 k')
            st.markdown('🔑 AI 解釋 Elbow method inertias (SSE)')
       
        with st.expander('PCA 主成分分析'):             
            st.markdown("""🔑 PCA (Principal Component Analysis)    
                **主成分分析**，是一種常見的**降維** (dimensionality reduction) 技術。主要目的在於將多個相關的變數，
                轉換成數量較少、且彼此不相關的**主成分** (principal components)。告訴你每個主成分是由哪些原始**特徵**所組成的，
                以及每個**特徵**的權重有多大。
                """)    
            st.markdown("""🔑 AI 解釋 PCA (Principal Component Analysis)""")        
        
        with st.expander('群集特徵'):
            st.markdown("""🔑 將**分群**結果與原始**特徵**結合，提供一個簡潔、高層次的概觀。這個表格的目的是幫助你快速理解每個**群集**的核心**特徵**""")
            st.markdown("""🔑 AI 解釋 群集特徵 Summary Table""")
            st.markdown("""🔑 列出各**群集**有多少檔股票**count**, 與原始**特徵**""")  
                         
        with st.expander('分群'):
            st.markdown("""🔑 表列出各**群集**的每一檔股票""")      
            st.markdown("""🔑 AI Agent 功能  
                任意點選一檔股票, 程式使用 **LangGraph** 透過 **TavilySearch** 搜尋工具搜尋該個股在網路上的相關資料, 
                再由在地端的 **Llama3.1** LLM 針對搜尋結果做總結。
                """) 
            
        st.info('✏️ 請在左側設定參數，然後按 🚀「載入並執行分析」!')

if __name__ == "__main__":
    main()
