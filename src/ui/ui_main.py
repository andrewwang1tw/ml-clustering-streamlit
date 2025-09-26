import platform
import streamlit as st
import matplotlib
import seaborn as sns
from agent.langGraph_gemini import stream_llm_updates

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
# 輔助函式：取得群集標籤列表
#-----------------------------------------------------------------------------------
def get_cluster_labels(df_clustered):
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    cluster_labels = [f"群 {i} ({cnt} 筆)" for i, cnt in cluster_counts.items()]
    return cluster_counts, cluster_labels

   
#-----------------------------------------------------------------------------------
# llm_chat
#-----------------------------------------------------------------------------------
def llm_chat(input: str):  
    with st.status("📂 AI 解釋中...", expanded=True) as status:       
        response = stream_llm_updates(input)     
        if response:                                  
            st.markdown(response) 
        
        status.update(label="📂 AI 解釋", state="complete", expanded=True)    
       

#-----------------------------------------------------------------------------------
# set_font 中文
#-----------------------------------------------------------------------------------       
def set_font():       
    sns.set_theme(style="whitegrid")  # 你也可以選 whitegrid, darkgrid, ticks 等    
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    if platform.system() == "Windows":
        matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
    elif platform.system() == "Darwin":
        matplotlib.rcParams['font.family'] = 'Heiti TC'
    else:
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

