from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator
import streamlit as st
import numpy as np
from data.db import make_engine, load_sales_data, load_sales_data_from_csv


#------------------------------------------------------------------------------------
# 1. 資料取得 
# 2. 特徵工程 
# 3. 相關分析, 去高度相關 
# 4. 標準化, 分群 
# 5. PCA 
# 6. 結果存到 session_state    
#------------------------------------------------------------------------------------
def run_analysis(params):    
    # 1. 取資料
    engine = make_engine(params['server'], params['database'], trusted=params['trusted'])
    df_raw = load_sales_data(engine, year=params['year'], month_from=params['month_from'])    
    
    #     
    #df_raw = load_sales_data_from_csv()
    #print("\n======> [run_analysis], df_raw ===>\n", df_raw)

    #--- ---
    df_wide = to_wide(df_raw)
    print("\n======> [run_analysis], df_wide ===>\n", df_wide)
    
    # 2. 特徵工程    
    mom_cols = sorted([c for c in df_wide.columns if c.startswith('MoM_')])
    yoy_cols = sorted([c for c in df_wide.columns if c.startswith('YoY_')]) 
    
    # --- continus_flag, 近 n 個月的 MoM/YoY 都 > 0 ---
    last_n = params['last_n_months']
    mom_sel = mom_cols[-last_n:] if len(mom_cols) >= last_n else mom_cols
    yoy_sel = yoy_cols[-last_n:] if len(yoy_cols) >= last_n else yoy_cols
    print("\n======> [run_analysis], mom_sel ===>\n", mom_sel)
    print("\n======> [run_analysis], yoy_sel ===>\n", yoy_sel)
        
    # --- 特徵工程 ---     
    df_feature = feature_engineer(df_wide, mom_sel, yoy_sel)
    print("\n======> [run_analysis], df_feature ===>\n", df_feature)

    # 3. 去高度相關前 和 去高度相關後 (但留 must_keep)  
    numeric_cols = df_feature.select_dtypes(include=['number']).columns.tolist()
    print("\n======> [run_analysis], numeric_cols ===>\n", numeric_cols)
    
    # --- 去高度相關前 --- 
    corr_before = df_feature[numeric_cols].corr()    
    
    # --- 去高度相關 ---
    must_keep = ['continuous_flag', 'avg_mom', 'std_mom', 'avg_yoy', 'std_yoy']
    df_reduced, dropped = remove_high_corr(df_feature, threshold=0.9, exclude_cols=must_keep)
    
    # --- 去高度相關後 (但留 must_keep)  ---    
    corr_after = df_reduced.select_dtypes(include=['number']).corr()

    # 4. 標準化後分群
    manual_k = params['manual_k']
    df_clustered, X_scaled, inertias, use_k, elbow_k = scale_and_cluster(
        df_reduced, 
        features=must_keep, 
        manual_k=(manual_k if manual_k > 0 else None),
        k_min=params['k_min'], 
        k_max=params['k_max']
    )
    print("\n======> [run_analysis], df_clustered, 標準化後分群 ===>\n", df_clustered)
    print("\n======> [run_analysis], X_scaled ===>\n", X_scaled)
    print("\n======> [run_analysis], inertias ===>\n", inertias)

    # 5. PCA 2D
    pca2, X_pca2 = run_pca(X_scaled, n_components=2)
    df_clustered['PC1'] = X_pca2[:,0]
    df_clustered['PC2'] = X_pca2[:,1]
    print("\n======> [run_analysis], df_clustered, PCA2 ===>\n", df_clustered)

    # 6. PCA 3D
    pca3, X_pca3 = run_pca(X_scaled, n_components=3)
    df_clustered['PC1_3'] = X_pca3[:,0]
    df_clustered['PC2_3'] = X_pca3[:,1]
    df_clustered['PC3_3'] = X_pca3[:,2]
    print("\n======> [run_analysis], df_clustered, PCA3 ===>\n", df_clustered)

    # 7. 最後存到 session_state
    st.session_state['df_raw'] = df_raw
    st.session_state['df_feature'] = df_feature
    st.session_state['df_clustered'] = df_clustered
    st.session_state['inertias'] = inertias
    st.session_state['elbow_k'] = elbow_k
    st.session_state['use_k'] = use_k
    st.session_state['corr_before'] = corr_before
    st.session_state['corr_after'] = corr_after
    st.session_state['features'] = must_keep
    st.session_state['dropped'] = dropped
    st.session_state['pca2'] = pca2    
    st.session_state['pca3'] = pca3   
    st.session_state['params'] = params    
    
    print("\n======> [run_analysis], copmpleted !!! \n")


#-----------------------------------------------------------------------------------
# 標準化, 分群(MiniBatchKMeans)
#-----------------------------------------------------------------------------------
@st.cache_data
def scale_and_cluster(df_reduced, features, manual_k=None, k_min=1, k_max=10):
    print("\n======> [run_analysis], manual_k ===>\n", manual_k)
    
    df = df_reduced.copy()  # ✅ 避免副作用
    if df.empty or len(df) == 0:
        raise ValueError("輸入的資料 df 是空的，請檢查前處理或資料來源")
    
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    for kk in range(k_min, k_max + 1):
        mbk = MiniBatchKMeans(n_clusters=kk, random_state=42, n_init=10, batch_size=256)
        mbk.fit(X_scaled)
        inertias.append(mbk.inertia_)

    elbow_k = None
    try:
        kl = KneeLocator(range(k_min, k_max + 1), inertias, curve='convex', direction='decreasing')
        elbow_k = kl.elbow
        print("\n======> [run_analysis], elbow_k ===>\n", elbow_k)
        
    except Exception:
        elbow_k = None

    use_k = manual_k if (manual_k is not None and manual_k > 0) else (elbow_k or 3)
    print("\n======> [run_analysis], use_k ===>\n", use_k)
    
    final_km = MiniBatchKMeans(n_clusters=use_k, random_state=42, n_init=10, batch_size=256)
    clusters = final_km.fit_predict(X_scaled)
    df2 = df.copy()
    df2['cluster'] = clusters

    return df2, X_scaled, inertias, use_k, elbow_k

#-----------------------------------------------------------------------------------
# PCA 
#-----------------------------------------------------------------------------------
@st.cache_data  
def run_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


#-----------------------------------------------------------------------------------
# 特徵工程
#-----------------------------------------------------------------------------------
@st.cache_data
def feature_engineer(df_wide, mom_months, yoy_months):
    df = df_wide.copy()  # ✅ 避免副作用
    
    # 確保所有要用的欄位存在
    for col in mom_months + yoy_months:
        if col not in df.columns:
            df[col] = np.nan

    def check_growth(row):
        mom_ok = all(row.get(c, np.nan) > 0 for c in mom_months)
        yoy_ok = all(row.get(c, np.nan) > 0 for c in yoy_months)
        return 1 if (mom_ok and yoy_ok) else 0

    # Add 特徵, continuous_flag 
    df['continuous_flag'] = df.apply(check_growth, axis=1)

    # Add MoM 特徵
    mom_cols = [c for c in df.columns if c.startswith('MoM_')]
    df['avg_mom'] = df[mom_cols].mean(axis=1)
    df['std_mom'] = df[mom_cols].std(axis=1)

    # Add YoY 特徵
    yoy_cols = [c for c in df.columns if c.startswith('YoY_')]
    df['avg_yoy'] = df[yoy_cols].mean(axis=1)
    df['std_yoy'] = df[yoy_cols].std(axis=1)

    # 把 NaN 填 0 或合理值
    df[['continuous_flag','avg_mom','std_mom','avg_yoy','std_yoy']] = \
        df[['continuous_flag','avg_mom','std_mom','avg_yoy','std_yoy']].fillna(0)

    return df

#---------------------------------------------------------------
# 去高度相關
#---------------------------------------------------------------
def remove_high_corr(df, threshold=0.9, exclude_cols=None):  
    if exclude_cols is None:
        exclude_cols = []
    
    to_drop = set()    
    numeric = df.select_dtypes(include=['number'])
    corr = numeric.corr().abs()    
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iloc[i, j] > threshold:
                col_j = cols[j]
                if col_j not in exclude_cols:
                    to_drop.add(col_j)

    df2 = df.drop(columns=list(to_drop), errors='ignore')
    return df2, to_drop

#-----------------------------------------------------------------------------------
# 
#-----------------------------------------------------------------------------------
def to_wide(df):
    df_pivot = df.pivot(index=['stock_id', 'stock_name'], columns='YM', values=['MoM', 'YoY'])
    df_wide = df_pivot.reset_index()
    
    # flatten multiindex columns
    df_wide.columns = ['_'.join(map(str, col)).strip() for col in df_wide.columns.values]
    return df_wide
