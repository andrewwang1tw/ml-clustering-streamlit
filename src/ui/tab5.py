import io
import json
import streamlit as st
import pandas as pd
from agent.langGraph_gemini import stream_graph_updates2, stream_graph_updates
from ui.ui_main import get_cluster_labels


#-----------------------------------------------------------------------------------
# Tab 5, show_cluster_details
#-----------------------------------------------------------------------------------
def tab5_show_cluster_details(df_clustered, page_size=10):    
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
            Summary Taiwan stock {str(selected_id)}.TW for the most recent 6 months basic information, stock price, company operation and related news.             
            Include domain will be https://tw.stock.yahoo.com/ , https://www.cnyes.com/ , https://www.ctee.com.tw/ and https://money.udn.com/ .            
            If you can not find related information just say "no information found!" directly.             
            Finally, must answer in Traditional Chinese.                                              
            """  
            
        # with st.status("🌐 LangGraph 搜尋中...", expanded=True) as status:              

        #     # 2. 建立一個聊天訊息區塊，顯示 AI 的回應
        #     with st.chat_message("assistant"):
        #         # st.write_stream 會接收 stream_graph_updates 產出的內容並即時顯示
        #         # response_generator 是我們上面修改後的 generator
        #         response_generator = stream_graph_updates2(user_input)
                
        #         # *** 關鍵修改：使用 st.write_stream 處理 generator ***
        #         full_response = st.write_stream(response_generator)
                
        #     # 3. 狀態完成 (可以選擇性地隱藏 status，或保持顯示)
        #     status.update(label="📂 AI 解釋", state="complete", expanded=True)
            
        #     # 備註：如果你需要保留完整的 full_response (例如用於快取)，
        #     # st.write_stream() 會回傳所有片段拼接起來的完整字串。
            
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
        
    