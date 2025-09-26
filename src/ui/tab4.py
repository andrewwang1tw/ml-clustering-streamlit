import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from ui.ui_main import llm_chat

#-----------------------------------------------------------------------------------
# Tab 41, show_cluster_summary
#-----------------------------------------------------------------------------------
def tab4_show_cluster_summary(df_clustered, features):
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
        llm_chat(input)  
            
    st.divider()


#-----------------------------------------------------------------------------------
# Tab 42, show_cluster_feature_boxplots
#-----------------------------------------------------------------------------------
def tab4_show_cluster_feature_boxplots(df_clustered, last_n_months):
    print("\ndf_clustered, Tab 42 ===>\n", df_clustered)    
    st.write('📋 **各群特徵差異**')
    
    #
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(data=df_clustered, x='cluster', y='continuous_flag', estimator='mean', ax=ax)
    
    #ax.set_title(f'最近{last_n_months}個月,月增與年增同時成長', fontsize=10)
    #ax.set_ylabel('月增/年增同時連續成長', fontsize=10)
    ax.set_title(f'mom/yoy growth for recent {last_n_months} months', fontsize=10)
    ax.set_ylabel('mom/yoy growth', fontsize=10)
    ax.set_xlabel('cluster', fontsize=10)    
    
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    st.pyplot(fig)

    #
    #for col, title in [('avg_mom','平均月增成長'), ('std_mom','月增標準差'), ('avg_yoy','平均年增成長'), ('std_yoy','年增標準差')]:
    for col, title in [('avg_mom','avg_mom'), ('std_mom','std_mom'), ('avg_yoy','avg_yoy'), ('std_yoy','std_yoy')]:
        fig_c, ax_c = plt.subplots(figsize=(6,3))
        sns.boxplot(data=df_clustered, x='cluster', y=col, ax=ax_c)
        #
        ax_c.set_title(f'{title}', fontsize=10)
        ax_c.set_ylabel(f'{title}', fontsize=10)
        ax_c.set_xlabel('cluster', fontsize=10)        
        #
        ax_c.tick_params(axis='x', labelsize=8)
        ax_c.tick_params(axis='y', labelsize=8)
        st.pyplot(fig_c)
