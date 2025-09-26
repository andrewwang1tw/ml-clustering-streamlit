import streamlit as st
from ui.ui_main import get_cluster_labels
                           
#-----------------------------------------------------------------------------------
# Tab 6, show_cluster_mom_yoy
#-----------------------------------------------------------------------------------
def tab6_show_cluster_mom_yoy(df_clustered, page_size=10):
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
    mom_cols_all = sorted([c for c in df_clustered.columns if c.startswith('MoM_')])
    if len(cluster_df) > 0 and mom_cols_all:
        st.write('📈 **MoM 時間序列**')
        mom_data = cluster_df[mom_cols_all].T
        mom_data.columns = cluster_df['stock_id_'].astype(str) + ' ' + cluster_df['stock_name_'].astype(str)
        st.line_chart(mom_data)
        
    # YoY 時序
    yoy_cols_all = sorted([c for c in df_clustered.columns if c.startswith('YoY_')])
    if len(cluster_df) > 0 and yoy_cols_all:
        st.write('📈 **YoY 時間序列**')
        yoy_data = cluster_df[yoy_cols_all].T
        yoy_data.columns = cluster_df['stock_id_'].astype(str) + ' ' + cluster_df['stock_name_'].astype(str)
        st.line_chart(yoy_data)

