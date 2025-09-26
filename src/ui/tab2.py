import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    st.pyplot(fig2)
   
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
  
 