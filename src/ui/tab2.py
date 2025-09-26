import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Optional
from ui.ui_main import llm_chat
      
#-----------------------------------------------------------------------------------
# Tab 2, show_elbow_plot
#-----------------------------------------------------------------------------------
def tab2_show_elbow_plot(k_min, k_max, inertias, use_k, elbow_k):  
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
    
    #st.pyplot(fig2)         
    st.plotly_chart(fig2, use_container_width=True)
   
    # --- AI  ---      
    input = f"""
            Let's say, the inertias: {inertias}, is the result from MiniBatchKMeans. 
            It's the mbk.inertia_ from mbk = MiniBatchKMeans(n_clusters=kk, random_state=42, n_init=10, batch_size=256) .                
            Please explain the inertias and list the full inertias. And answer in Traditional Chinese.  
            """            
    if st.button("AI 解釋 Elbow method inertias (SSE)", key="ai_elbow_method_elbow method"):
        #with st.status("📂 AI 解釋中...", expanded=True) as status: 
            llm_chat(input) 
            #status.update(label="📂 AI 解釋", state="complete", expanded=True) 
   
    #
    st.dataframe(pd.DataFrame(inertias, columns=["inertias, Sum of Squared Errors (SSE)"]))
  
 
#-----------------------------------------------------------------------------------
# Tab 2, show_elbow_plot2
#-----------------------------------------------------------------------------------
def tab2_show_elbow_plot2(k_min: int, k_max: int, inertias: List[float], use_k: int, elbow_k: Optional[int]):
    """
    使用 Plotly Express 顯示 K-Means 肘部法則圖，並標註建議的 K 值。

    Args:
        k_min (int): K 值的起始範圍。
        k_max (int): K 值的結束範圍。
        inertias (List[float]): 對應於每個 K 值的 SSE (Inertia) 列表。
        use_k (int): 使用者當前選擇的分群數。
        elbow_k (Optional[int]): 透過 Kneelocator 找到的建議「手肘」K 值。
    """
    
    # ----------------------------------------------------
    # 1. 顯示標題和建議 K 值
    # ----------------------------------------------------
    st.markdown(f"📉 **目前分群 (k):** `{use_k}` | **最佳分群 (Elbow K):** `{elbow_k if elbow_k is not None else 'N/A'}`")

    # 2. 將數據轉換為 Pandas DataFrame (Plotly Express 的推薦做法)
    k_range = list(range(k_min, k_max + 1))
    
    # 確保 inertias 的長度與 k_range 匹配，否則程式會崩潰
    if len(k_range) != len(inertias):
        st.error("錯誤：K 值範圍與 Inertia 列表長度不匹配。")
        return

    df = pd.DataFrame({
        'Number of Clusters (k)': k_range,
        'Inertia (SSE)': inertias,
        # 建立一個欄位來標記 Elbow Point，方便在圖表上凸顯
        'Highlight': [f'Elbow K={elbow_k}' if k == elbow_k else 'Other K' for k in k_range]
    })
    
    # ----------------------------------------------------
    # 3. 使用 Plotly Express 繪製圖表
    # ----------------------------------------------------
    fig2 = px.scatter(
        df,
        x='Number of Clusters (k)',
        y='Inertia (SSE)',
        color='Highlight', 
        # 設置顏色：Elbow K 用紅色，其他用藍色/灰色
        color_discrete_map={f'Elbow K={elbow_k}': 'red', 'Other K': 'blue'},
        title='Elbow Method (SSE vs. Number of Clusters)',
        hover_name='Number of Clusters (k)' # 滑鼠懸停時顯示 K 值
    )

    # ----------------------------------------------------
    # 4. 關鍵步驟：強制顯示線和點，並調整佈局
    # ----------------------------------------------------
    
    # 確保同時顯示線和點 (這是 Elbow Plot 必要的視覺呈現)
    fig2.update_traces(
        mode='lines+markers',
        marker=dict(size=8),
        # 對於非 Elbow K 的點，可以讓它們顏色變淡
        selector=dict(name='Other K') 
    )

    # 調整佈局
    fig2.update_layout(
        # 調整標題和邊距
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor='rgba(245,245,245,1)', 
        # 讓滑鼠懸停資訊更簡潔，並將圖例放在底部
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # ----------------------------------------------------
    # 5. 標註 Elbow Point (使用 annotation)
    # ----------------------------------------------------
    if elbow_k in k_range:
        elbow_index = df[df['Number of Clusters (k)'] == elbow_k].index[0]
        
        fig2.add_annotation(
            x=elbow_k,
            y=df.loc[elbow_index, 'Inertia (SSE)'],
            text=f"Optimal K={elbow_k}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            font=dict(color="red", size=14, family="Arial"),
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            bgcolor="#ffeedd",
            opacity=0.8
        )
    
    # ----------------------------------------------------
    # 6. 顯示在 Streamlit 介面
    # ----------------------------------------------------
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- AI  ---      
    input = f"""
            Let's say, the inertias: {inertias}, is the result from MiniBatchKMeans. 
            It's the mbk.inertia_ from mbk = MiniBatchKMeans(n_clusters=kk, random_state=42, n_init=10, batch_size=256) .                
            Please explain the inertias and list the full inertias. And answer in Traditional Chinese.  
            """            
    if st.button("AI 解釋 Elbow method inertias (SSE)", key="ai_elbow_method_elbow method"):
        #with st.status("📂 AI 解釋中...", expanded=True) as status: 
            llm_chat(input) 
            #status.update(label="📂 AI 解釋", state="complete", expanded=True) 
   
    st.dataframe(pd.DataFrame(inertias, columns=["inertias, Sum of Squared Errors (SSE)"]))