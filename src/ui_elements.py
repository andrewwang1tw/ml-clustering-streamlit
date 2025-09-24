import io
import json
import platform
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from agent.langGraph_ollama import stream_graph_updates, stream_ollama_updates
import base64
import matplotlib.font_manager as fm

#-----------------------------------------------------------------------------------
# 顯示 sidebar 輸入元件，回傳所有參數值
#-----------------------------------------------------------------------------------
def sidebar_inputs():  
    # 在 sidebar 建立一個容器，讓按鈕顯示在最上面
    top_container = st.sidebar.container()
    run = top_container.button('🚀 載入並執行分析', key='run_button')
 
    st.sidebar.header('查詢條件')
    year = st.sidebar.number_input('民國年 (年)', min_value=100, max_value=200, value=114, key="query_year")
    month_from = st.sidebar.number_input('起始月 (>=1)', min_value=1, max_value=12, value=1, key="query_month_from")

    st.sidebar.header('Feature 選項')
    last_n_months = st.sidebar.slider('最近 n 個月營收月增與年增均成長', 1, 12, 3, key="feature_n_months")

    st.sidebar.header('Clustering 選項')
    k_min = st.sidebar.number_input('k 最低', min_value=1, max_value=10, value=1, key="k_min")
    k_max = st.sidebar.number_input('k 最高', min_value=2, max_value=20, value=10, key="k_max")
    manual_k = st.sidebar.number_input('手動設定 k (0=自動)', min_value=0, max_value=10, value=3, key="manual_k")
    
    st.sidebar.header('資料庫連線')
    server = st.sidebar.text_input('SQL Server', value='localhost', key="db_server")
    database = st.sidebar.text_input('Database', value='stock_tw', key="db_name")
    trusted = st.sidebar.checkbox('Use Trusted Connection (Windows Auth)', value=True, key="trusted_connection")

    return {
        'server': server,
        'database': database,
        'trusted': trusted,
        'year': year,
        'month_from': month_from,
        'last_n_months': last_n_months,
        'k_min': k_min,
        'k_max': k_max,
        'manual_k': manual_k,
        'run': run
    }


#-----------------------------------------------------------------------------------
# Tab 0, 
#-----------------------------------------------------------------------------------
def show_sales_data(df_raw, df_feature):
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
        

#-----------------------------------------------------------------------------------
# Tab 1, show_correlation_heatmap
#-----------------------------------------------------------------------------------
def show_correlation_heatmap(corr_before, corr_after, dropped=None):
    print("\ncorr_before, Tab 1 ===>\n", corr_before)
       
    option = st.selectbox("📊 選擇", ["Before, 去高度相關前", "After, 去高度相關後"], key='select_corr_option')
    corr_df = corr_before if option.startswith("Before") else corr_after   
              
    #st.write(title)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr_df, annot=False, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    st.pyplot(fig)

    # 顯示高相關對
    st.write("⚠️ 高相關特徵 Pairs")
    threshold = 0.9
    high_corr_pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_df.iloc[i, j] > threshold:
                high_corr_pairs.append((cols[i], cols[j], round(corr_df.iloc[i, j], 3)))

    if high_corr_pairs:
        st.dataframe(pd.DataFrame(high_corr_pairs, columns=["Feature A","Feature B","Correlation"]))
    else:
        st.success("沒有超過相關係數閾值的特徵 pair ✅")
        
    # 被移除的高相關特徵
    if dropped is not None:
        st.write("🗑️ 被移除的高相關特徵")
        st.write(dropped if dropped else "✅ 沒有特徵被移除")    


#-----------------------------------------------------------------------------------
# ollama_chat
#-----------------------------------------------------------------------------------
def ollama_chat(input: str):  
    with st.status("📂 AI 解釋中...", expanded=True) as status:       
        response = stream_ollama_updates(input)     
        if response:                                  
            st.markdown(response) 
        
        status.update(label="📂 AI 解釋", state="complete", expanded=True)    
       

#-----------------------------------------------------------------------------------
# Tab 2, show_elbow_plot
#-----------------------------------------------------------------------------------
def show_elbow_plot(k_min, k_max, inertias, use_k, elbow_k):  
    print("\nuse_k, Tab 2 ===>\n", use_k)
    print("\ninertias, Tab 2 ===>\n", inertias)
    
    #st.markdown('📉 **Elbow Plot**')    
    st.markdown(f"📉 **目前分群:** {use_k}, **最佳分群 (KneeLocator elbow point):** {elbow_k} ")

    # Elbow Method
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(range(k_min, k_max+1), inertias, marker='o')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Inertia (SSE)')
    ax2.set_title('Elbow Method')
    st.pyplot(fig2)
   
    # --- AI  ---      
    input = f"""
            Let's say, the inertias: {inertias}, is the result from MiniBatchKMeans. 
            It's the mbk.inertia_ from mbk = MiniBatchKMeans(n_clusters=kk, random_state=42, n_init=10, batch_size=256) .                
            Please explain the inertias and list the full inertias. And answer in Traditional Chinese.  
            """            
    if st.button("AI 解釋 Elbow method inertias (SSE)", key="ai_elbow_method_elbow method"):
        #with st.status("📂 AI 解釋中...", expanded=True) as status: 
            ollama_chat(input) 
            #status.update(label="📂 AI 解釋", state="complete", expanded=True) 
   
    #
    st.dataframe(pd.DataFrame(inertias, columns=["inertias, Sum of Squared Errors (SSE)"]))
  
 
#-----------------------------------------------------------------------------------
# Tab 31, show_pca
#-----------------------------------------------------------------------------------
def show_pca_3d(df_clustered, pca3, features):
    print("\ndf_clustered, Tab 31 ===>\n", df_clustered)
    print("\npca 3d, Tab 31 ===>\n", pca3)
    print("\nfeatures, Tab 31 ===>\n", features)
    
    st.write("🔵 主成分分析 (PCA 3D)")    
    fig4 = px.scatter_3d(df_clustered, x='PC1_3', y='PC2_3', z='PC3_3',
                         color=df_clustered['cluster'].astype(str),
                         hover_data=['stock_id_', 'stock_name_'])   
     
    fig4.update_layout(title='PCA 3D 分群', legend_title='cluster',
                       paper_bgcolor='rgba(250,250,250,1)', margin=dict(l=40, r=40, t=40, b=40))  
      
    st.plotly_chart(fig4, use_container_width=True)

    # --- 各主成分變異比例 ---    
    st.write("**各主成分變異比例 (Explained Variance Ratio)**") 
    df = pd.DataFrame({
        "PC1": [pca3.explained_variance_ratio_[0]], 
        "PC2": [pca3.explained_variance_ratio_[1]], 
        "PC3": [pca3.explained_variance_ratio_[2]], 
        "總解釋比例": [sum(pca3.explained_variance_ratio_[:3])]
    })    
    styled_df = df.style.format("{:.4f}")
    st.dataframe(styled_df, hide_index=True)
   
    # --- 主成分組成 ---
    st.write("**主成分組成 (Components)**")
    comp_df3 = pd.DataFrame(pca3.components_, columns=features, index=["PC1", "PC2", "PC3"])
    st.dataframe(comp_df3.style.format("{:.3f}"))
    print("\ncomp_df3, Tab 31 ===>\n", comp_df3)
      
    # --- AI  ---      
    input = f"""
                Please explain the 3D PCA(Principal Component Analysis) analysis with 
                Explained Variance Ratio: {df},
                Components:{comp_df3}, 
                Features as below
                continuous_flag : 近幾個月連續營收月增率與年增率均成長時為 1. 
                avg_mom : 平均月增率, 反映公司成長動能.
                std_mom : 月增率的標準差, 反映公司營收的波動性.
                avg_yoy : 平均年增率, 反映公司成長動能.
                std_yoy : 年增率的標準差, 反映公司營收的波動性.
                And answer in Traditional Chinese.
            """            
    if st.button("AI 解釋 PCA (3D)", key="ai_pca_3d"):       
        ollama_chat(input)
      
         
    st.divider()
    
#-----------------------------------------------------------------------------------
# Tab 32, show_pca
#-----------------------------------------------------------------------------------
def show_pca_2d(df_clustered, pca2, features):
    print("\ndf_clustered, Tab 32 ===>\n", df_clustered)
    print("\npca 2d, Tab 32 ===>\n", pca2)
    print("\nfeatures, Tab 32 ===>\n", features)
    
    st.write("🔵 主成分分析 (PCA 2D)")
    
    fig3 = px.scatter(df_clustered, x='PC1', y='PC2',
                      color=df_clustered['cluster'].astype(str),
                      hover_data=['stock_id_', 'stock_name_'])
    
    fig3.update_layout(title='PCA 2D 分群', legend_title='cluster',
                       paper_bgcolor='rgba(250,250,250,1)', margin=dict(l=40, r=40, t=40, b=40))
    
    st.plotly_chart(fig3, use_container_width=True)

    # --- 各主成分變異比例 ---
    st.write("**各主成分變異比例 (Explained Variance Ratio)**") 
    df = pd.DataFrame({
        "PC1": [pca2.explained_variance_ratio_[0]], 
        "PC2": [pca2.explained_variance_ratio_[1]], 
        "總解釋比例": [sum(pca2.explained_variance_ratio_[:2])]
    })    
    styled_df = df.style.format("{:.4f}")
    st.dataframe(styled_df, hide_index=True)

    # --- 主成分組成 ---
    st.write("**主成分組成 (Components)**")
    comp_df = pd.DataFrame(pca2.components_, columns=features, index=["PC1", "PC2"])
    st.dataframe(comp_df.style.format("{:.3f}"))
    
    # --- AI ---
    input = f"""
                Please explain the 2D PCA(Principal Component Analysis) analysis with 
                Explained Variance Ratio: {df},
                Components:{comp_df}, 
                Features as below
                continuous_flag : 近幾個月連續營收月增率與年增率均成長時為 1. 
                avg_mom : 平均月增率, 反映公司成長動能.
                std_mom : 月增率的標準差, 反映公司營收的波動性.
                avg_yoy : 平均年增率, 反映公司成長動能.
                std_yoy : 年增率的標準差, 反映公司營收的波動性.
                And answer in Traditional Chinese.
            """            
    if st.button("AI 解釋 PCA (2D)", key="ai_pca_2d"):
        ollama_chat(input)  
            
    st.divider()
    
#-----------------------------------------------------------------------------------
# Tab 41, show_cluster_summary
#-----------------------------------------------------------------------------------
def show_cluster_summary(df_clustered, features):
    print("\ndf_clustered, Tab 41 ===>\n", df_clustered)
    print("\nfeatures, Tab 41 ===>\n", features)        
    st.write('📋 **群集特徵 Summary Table**')
    
    summary = df_clustered.groupby('cluster').agg({
        'stock_id_': 'count',
        'continuous_flag': 'mean',
        'avg_mom': 'mean',
        'std_mom': 'mean',
        'avg_yoy': 'mean',
        'std_yoy': 'mean'
    }).rename(columns={'stock_id_': 'count'})
        
    # 熱度背景格式化        
    styled = summary.style.format({
        'count':"{:.0f}",
        'continuous_flag':"{:.0f}",
        'avg_mom': "{:.1f}",
        'std_mom': "{:.1f}",
        'avg_yoy': "{:.1f}",
        'std_yoy': "{:.1f}"
    }).background_gradient(
        cmap="RdYlGn",  # 雙向顏色 (紅→黃→綠)
        axis=None,      # 整張表統一顏色比例
        vmin=-summary.abs().max().max(),  # 負到正對稱
        vmax=summary.abs().max().max()
    )
    
    st.dataframe(styled)
    
    # --- AI ---
    input = f"""
                Please explain the machine learning cluster, summary as {summary} and features {features}
                Features as below
                count : the  number of the stock in the cluster.
                continuous_flag : 近幾個月的每月營收月增率與年增率同時都正成長時為 1, 否則 0. 
                avg_mom : 平均月增率, 反映公司成長動能.
                std_mom : 月增率的標準差, 反映公司營收的波動性.
                avg_yoy : 平均年增率, 反映公司成長動能.
                std_yoy : 年增率的標準差, 反映公司營收的波動性.                
                And answer in Traditional Chinese.
            """            
    if st.button("AI 群集特徵 Summary Table", key="ai_cluster_summary"):
        ollama_chat(input)  
            
    st.divider()


#-----------------------------------------------------------------------------------
# Tab 42, show_cluster_feature_boxplots
#-----------------------------------------------------------------------------------
def show_cluster_feature_boxplots(df_clustered, last_n_months):
    print("\ndf_clustered, Tab 42 ===>\n", df_clustered)    
    st.write('📋 **各群特徵差異**')
    
    #
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(data=df_clustered, x='cluster', y='continuous_flag', estimator='mean', ax=ax)
    ax.set_title(f'最近{last_n_months}個月,月增與年增同時成長', fontsize=10)
    ax.set_xlabel('群集(cluster)', fontsize=10)
    ax.set_ylabel('月增/年增同時連續成長', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)

    #
    for col, title in [('avg_mom','平均月增成長'), ('std_mom','月增標準差'),
                       ('avg_yoy','平均年增成長'), ('std_yoy','年增標準差')]:
        fig_c, ax_c = plt.subplots(figsize=(6,3))
        sns.boxplot(data=df_clustered, x='cluster', y=col, ax=ax_c)
        ax_c.set_title(f'{title}', fontsize=10)
        ax_c.set_xlabel('群集(cluster)', fontsize=10)
        ax_c.set_ylabel(f'{title}', fontsize=10)
        ax_c.tick_params(axis='x', labelsize=8)
        ax_c.tick_params(axis='y', labelsize=8)
        st.pyplot(fig_c)

#-----------------------------------------------------------------------------------
# 輔助函式：取得群集標籤列表
#-----------------------------------------------------------------------------------
def get_cluster_labels(df_clustered):
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    cluster_labels = [f"群 {i} ({cnt} 筆)" for i, cnt in cluster_counts.items()]
    return cluster_counts, cluster_labels

#-----------------------------------------------------------------------------------
# Tab 5, show_cluster_details
#-----------------------------------------------------------------------------------
def show_cluster_details(df_clustered, page_size=10):    
    #st.subheader('🗓️ 群集詳細檢視')
    print("\ndf_clustered, Tab 5 ===>\n", df_clustered)    
        
    # --- 每一個 cluster 的筆數 ---   
    cluster_counts, cluster_labels = get_cluster_labels(df_clustered)
    col1, col2 = st.columns([2, 1])
    with col1: 
        selected = st.selectbox('📈 選擇群集', options=cluster_labels, key="cluster_select_label")
    
    # --- 找出所選的 cluster ---
    idx = cluster_labels.index(selected)
    cluster_index = cluster_counts.index[idx]
    df_cluster_selected = df_clustered[df_clustered['cluster'] == cluster_index]
    print("\ndf_cluster_selected, Tab 5 ===>\n", df_cluster_selected)

    total = len(df_cluster_selected)
    total_pages = (total - 1) // page_size + 1 if total > 0 else 1         
    with col2: 
        page = st.number_input("📈 頁碼", min_value=1, max_value=total_pages, value=1, step=1, key="cluster_page_num")    
        
    start = (page - 1) * page_size
    end = start + page_size        
    st.write(f'📈 群 {cluster_index} (count={total}) - 第 {page}/{total_pages} 頁')
    
    df = df_cluster_selected[['stock_id_','stock_name_', 
                    'continuous_flag',
                    'avg_mom','std_mom', 
                    'avg_yoy','std_yoy']].iloc[start:end].copy()    
     
    selected_rows_dict = st.dataframe(
        df,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"        
    )   
    
    # --- 提供下載該群 Excel ---
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_cluster_selected.to_excel(writer, index=False, sheet_name=f'cluster_{cluster_index}')
        
    excel_data = buffer.getvalue()
    st.download_button(
        label="📈 下載該群結果 Excel",
        data=excel_data,
        file_name=f"cluster_{cluster_index}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # --- 取得選取到的資料, 如果有任何一行被選取 ---
    selected_indices = selected_rows_dict.get('selection', {}).get('rows', [])
    if selected_indices:
        # 取得被選取的列的索引
        print("\n選取到的資料, selected_indices ===>\n", selected_indices, "\n")
        selected_row_index = selected_indices[0]       
        selected_row_data = df.iloc[selected_row_index]        
        print("\n選取到的資料m selected_row_data ===>\n", selected_row_data, "\n")
        
        # 取得特定欄位的值
        selected_id = selected_row_data['stock_id_']
        selected_name = selected_row_data['stock_name_']
        print("\n取得選取到的資料 ===>\nselected_id= ", selected_id, ", selected_name= ", selected_name, "\n")
        
        # --- Use LangGraph --- 
        user_input = f"""
            Summary Taiwan stock {str(selected_id)}.TW for basic information and recent stock price, company operation and related news.             
            Include domain will be https://tw.stock.yahoo.com/ , https://www.cnyes.com/ , https://www.ctee.com.tw/ and https://money.udn.com/ .            
            If you can not find related information just say "no information found!" directly.             
            Finally, must answer in Traditional Chinese.                                              
            """  
        with st.status("🌐 LangGraph 搜尋中...", expanded=True) as status:              
            all_messages, last_msg = stream_graph_updates(user_input)                            
            if all_messages:
                status.update(label="🌐 LangGraph 任務已完成！", state="complete", expanded=True)                      
                for message in all_messages: 
                    try:        
                        data = json.loads(message)  
                        if data:
                            with st.expander("📂 AI Agent 透過 TavilySearch 搜尋到的資料"):                                     
                                for result in data.get('results', []):
                                    with st.container():
                                        st.markdown(f"**[{result.get('title', '無標題')}]({result.get('url', '#')})**")
                                        st.markdown(f"*{result.get('content', '無摘要')}*")
                                    
                    except json.JSONDecodeError:
                        # 如果不是有效的 JSON，就當作普通文字顯示    
                        with st.expander("📦 AI Agent 總結搜尋到的資料"):
                            st.write(message)
        
                           
#-----------------------------------------------------------------------------------
# Tab 6, show_cluster_mom_yoy
#-----------------------------------------------------------------------------------
def show_cluster_mom_yoy(df_clustered, page_size=10):
    #st.subheader('📈 MoM, YoY')
    print("\ndf_clustered, Tab 6 ===>\n", df_clustered)
      
    # 
    cluster_counts, cluster_labels = get_cluster_labels(df_clustered)
    selected = st.selectbox('選擇群集', options=cluster_labels, key="cluster_select_label2")
    
    # 找出所選的 cluster
    idx = cluster_labels.index(selected)
    cluster_index = cluster_counts.index[idx]
    cluster_df = df_clustered[df_clustered['cluster'] == cluster_index]
    
    # MoM 時序
    mom_cols_all = sorted([c for c in df_clustered.columns if c.startswith('月增_')])
    if len(cluster_df) > 0 and mom_cols_all:
        st.write('📈 **MoM 時間序列**')
        mom_data = cluster_df[mom_cols_all].T
        mom_data.columns = cluster_df['stock_id_'].astype(str) + ' ' + cluster_df['stock_name_'].astype(str)
        st.line_chart(mom_data)
        
    # YoY 時序
    yoy_cols_all = sorted([c for c in df_clustered.columns if c.startswith('年增_')])
    if len(cluster_df) > 0 and yoy_cols_all:
        st.write('📈 **YoY 時間序列**')
        yoy_data = cluster_df[yoy_cols_all].T
        yoy_data.columns = cluster_df['stock_id_'].astype(str) + ' ' + cluster_df['stock_name_'].astype(str)
        st.line_chart(yoy_data)


#-----------------------------------------------------------------------------------
# set_font 中文
#-----------------------------------------------------------------------------------       
def set_font():       
    # sns.set_theme(style="whitegrid")  # 你也可以選 whitegrid, darkgrid, ticks 等    
    # matplotlib.rcParams['axes.unicode_minus'] = False
    
    # if platform.system() == "Windows":
    #     matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
    # elif platform.system() == "Darwin":
    #     matplotlib.rcParams['font.family'] = 'Heiti TC'
    # else:
    #     matplotlib.rcParams['font.family'] = 'Noto Sans CJK TC'  
    
    # Force Matplotlib to rebuild its font cache to detect newly installed fonts.
    fm.rebuild()

    sns.set_theme(style="whitegrid")    
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Check the operating system
    if platform.system() == "Windows":
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
    elif platform.system() == "Darwin":
        matplotlib.rcParams['font.family'] = 'Heiti TC'
    else:  # Assumed to be Linux, e.g., Streamlit Cloud
        # This will now correctly find the font because it's been installed and the cache is rebuilt.
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK TC'
    
#-----------------------------------------------------------------------------------
# set_layout
#-----------------------------------------------------------------------------------        
def set_layout():
    st.set_page_config(
        layout='centered', 
        page_title='📈 營收分群與 PCA 分析 (Streamlit)'
    )  
    
    # 浮水印的 CSS 樣式
    watermark_css = """
    <style>
    .watermark {
        position: fixed;
        top: 2%;
        left: 45%;
        #transform: translate(-50%, -50%) rotate(-40deg);
        font-size: 5rem;
        font-weight: bold;
        color: rgba(128, 128, 128, 0.1); /* 灰色的透明度 */
        user-select: none;
        pointer-events: none;
        z-index: 1000;
    }
    .watermark-img {{
        width: 200px; /* 圖片大小 */
        height: 100px;
    }}
    </style>
    """  
    
    # 寫入 CSS, 加入浮水印文字
    st.markdown(watermark_css, unsafe_allow_html=True)   
    st.markdown('<div class="watermark">Andrew Wang</div>', unsafe_allow_html=True)
    
    # # 加入 Image
    # image_path = "img/1.png"
    # def get_base64_image(path):
    #     with open(path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode()
    
    # image_b64 = get_base64_image(image_path)    
    # st.markdown(f'<div class="watermark"><img src="data:image/png;base64,{image_b64}" class="watermark-img"></div>', unsafe_allow_html=True)

