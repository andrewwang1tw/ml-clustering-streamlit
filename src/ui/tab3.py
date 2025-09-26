import streamlit as st
import pandas as pd
import plotly.express as px
from ui.ui_main import llm_chat


#-----------------------------------------------------------------------------------
# Tab 31, show_pca
#-----------------------------------------------------------------------------------
def tab3_show_pca_3d(df_clustered, pca3, features):
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
        llm_chat(input)
      
         
    st.divider()
    
#-----------------------------------------------------------------------------------
# Tab 32, show_pca
#-----------------------------------------------------------------------------------
def tab3_show_pca_2d(df_clustered, pca2, features):
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
        llm_chat(input)  
            
    st.divider()
  