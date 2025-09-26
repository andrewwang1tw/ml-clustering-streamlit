import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------------------------------
# Tab 1, show_correlation_heatmap
#-----------------------------------------------------------------------------------
def tab1_show_correlation_heatmap(corr_before, corr_after, dropped=None):
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
