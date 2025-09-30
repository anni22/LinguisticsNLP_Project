import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev


def format_speakers(count):
    if count >= 1_000_000_000:
        return f'{count / 1_000_000_000:.1f}B'
    elif count >= 1_000_000:
        return f'{count / 1_000_000:.1f}M'
    else:
        return f'{int(count):,}'

def display_comparison(title, df1, df2, key_prefix, show_filtered_graphs=True):
    st.subheader(title)
    
    if df1 is not None and df2 is not None:
        merged_df = pd.merge(df1, df2, on='language', how='outer')
        merged_df.rename(columns={'cluster_1': 'Our taxonomy', 'cluster_2': 'Joshi et al. (2020)'}, inplace=True)
        
        merged_df['cluster_changed'] = merged_df['Our taxonomy'].notna() & merged_df['Joshi et al. (2020)'].notna() & (merged_df['Our taxonomy'] != merged_df['Joshi et al. (2020)'])

        st.subheader(f"Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Languages in our taxonomy", len(df1))
        col2.metric("Languages in Joshi et al.'s (2020) taxonomy", len(df2))
        col3.metric("Common Languages", len(merged_df.dropna(subset=['Our taxonomy', 'Joshi et al. (2020)'])))
        col4.metric("Languages with Changed Clusters", len(merged_df[merged_df['cluster_changed']]))

        st.subheader(f"Cluster Distribution")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Joshi et al.'s (2020) taxonomy")
            cluster2_counts = df2['cluster_2'].value_counts().reset_index()
            cluster2_counts.columns = ['Cluster', 'Count']
            fig2 = px.bar(cluster2_counts, x='Cluster', y='Count', title='Languages per Cluster in Joshi et al.\'s (2020) taxonomy')
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_2")

        with col2:
            st.write("our taxonomy")
            cluster1_counts = df1['cluster_1'].value_counts().reset_index()
            cluster1_counts.columns = ['Cluster', 'Count']
            fig1 = px.bar(cluster1_counts, x='Cluster', y='Count', title='Languages per Cluster in our taxonomy')
            st.plotly_chart(fig1, use_container_width=True, key=f"{key_prefix}_1")

        if show_filtered_graphs:
            st.subheader(f"Cluster Distribution (excluding cluster 0)")
            col3, col4 = st.columns(2)

            with col3:
                st.write("Joshi et al.'s (2020) taxonomy (no 0)")
                df2_filtered = df2[df2['cluster_2'] != 0]
                cluster2_counts_filtered = df2_filtered['cluster_2'].value_counts().reset_index()
                cluster2_counts_filtered.columns = ['Cluster', 'Count']
                fig4 = px.bar(cluster2_counts_filtered, x='Cluster', y='Count', title="Languages per Cluster in Joshi et al.'s (no 0)")
                st.plotly_chart(fig4, use_container_width=True, key=f"{key_prefix}_4")

            with col4:
                st.write("Our taxonomy (no class 0)")
                df1_filtered = df1[df1['cluster_1'] != 0]
                cluster1_counts_filtered = df1_filtered['cluster_1'].value_counts().reset_index()
                cluster1_counts_filtered.columns = ['Cluster', 'Count']
                fig3 = px.bar(cluster1_counts_filtered, x='Cluster', y='Count', title='Languages per Cluster in our taxonomy (no class 0)')
                st.plotly_chart(fig3, use_container_width=True, key=f"{key_prefix}_3")

        st.subheader(f"Languages with Changed Clusters")
        changed_languages_df = merged_df[merged_df['cluster_changed']][['language', 'Our taxonomy', 'Joshi et al. (2020)']]
        st.dataframe(changed_languages_df)

        st.subheader("Analysis of Cluster Changes")

        changed_df = merged_df[merged_df['cluster_changed']].copy()
        changed_df['change_direction'] = np.select(
            [
                changed_df['Our taxonomy'] > changed_df['Joshi et al. (2020)'],
                changed_df['Our taxonomy'] < changed_df['Joshi et al. (2020)']
            ],
            [
                'Shifted Up',
                'Shifted Down'
            ],
            default='No Change'
        )

        changed_df['shift_amount'] = (changed_df['Our taxonomy'] - changed_df['Joshi et al. (2020)']).abs()

        change_summary = changed_df.groupby('change_direction').agg(
            Count=('language', 'size'),
            Average_Shift=('shift_amount', 'mean')
        ).reset_index()

        total_changed = change_summary['Count'].sum()
        change_summary['Percentage'] = (change_summary['Count'] / total_changed) * 100
        
        change_summary.rename(columns={'change_direction': 'Direction', 'Average_Shift': 'Average Cluster Shift'}, inplace=True)

        fig_change = px.bar(change_summary, x='Direction', y='Percentage', title='Direction of Cluster Changes')
        fig_change.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_change, use_container_width=True, key=f"{key_prefix}_change_dir")

        st.write("Summary of Cluster Changes:")
        st.dataframe(change_summary)

        st.write("Languages that shifted up (higher cluster in our taxonomy):")
        shifted_up_df = changed_df[changed_df['change_direction'] == 'Shifted Up'][['language', 'Joshi et al. (2020)', 'Our taxonomy']]
        st.dataframe(shifted_up_df)

st.set_page_config(layout="wide")

st.title('Language Resource Analysis Dashboard')

tab1, tab2, tab3 = st.tabs(["Our taxonomy (only languages with resources)", "Our taxonomy (all 7500 languages)", "Taxonomy Comparison (Full Names)"])

with tab1:
    st.header("Our taxonomy (only languages with resources)")

    try:
        df_res = pd.read_csv('resource_distribution_with_speakers.csv', delimiter=';')
        df_res['SPEAKER_COUNT'] = pd.to_numeric(df_res['SPEAKER_COUNT'], errors='coerce').fillna(0)

        try:
            taxonomy_df_res = pd.read_csv('taxonomy_original_tags_fullnames.csv', header=None)
            taxonomy_df_res.columns = ['Tag', 'Full Language Name', 'Cluster']
            df_res = pd.merge(df_res, taxonomy_df_res[['Full Language Name']], left_on='LANGUAGE', right_on='Full Language Name', how='left')
            df_res['Full Language Name'] = df_res['Full Language Name'].fillna(df_res['LANGUAGE'])
        except FileNotFoundError:
            st.warning('Warning: `taxonomy_original_tags_fullnames.csv` not found. Full language names will not be available.')
            df_res['LANGUAGE']

        df_res['log_unlabelled'] = np.log10(df_res['UNLABELLED RESOURCE COUNT'].replace(0, 1))
        df_res['log_labelled'] = np.log10(df_res['LABELLED RESOURCE COUNT'].replace(0, 1))

        X_res = df_res[['log_unlabelled', 'log_labelled']]
        kmeans_res = KMeans(n_clusters=6, random_state=42, n_init=10)
        df_res['cluster'] = kmeans_res.fit_predict(X_res)

        centers_res = kmeans_res.cluster_centers_
        center_sums_res = centers_res.sum(axis=1)
        sorted_indices_res = np.argsort(center_sums_res)
        label_mapping_res = {original_label: new_label for new_label, original_label in enumerate(sorted_indices_res)}
        df_res['cluster'] = df_res['cluster'].map(label_mapping_res)
        sorted_centers_res = centers_res[sorted_indices_res]

        cluster_agg_res = df_res.groupby('cluster').agg(
            language_count=('LANGUAGE', 'size'),
            total_speakers=('SPEAKER_COUNT', 'sum')
        ).reset_index()

        cluster_agg_res['log_unlabelled'] = [center[0] for center in sorted_centers_res]
        cluster_agg_res['log_labelled'] = [center[1] for center in sorted_centers_res]

        cluster_colors_res = ['#6A5ACD', '#483D8B', '#6495ED', '#90EE90', '#FFD700', '#FFA07A', '#CD5C5C']

        fig_cluster_res = go.Figure()
        fig_cluster_res.add_trace(go.Scatter(
            x=df_res['log_unlabelled'],
            y=df_res['log_labelled'],
            mode='markers',
            marker=dict(color='lightgrey', size=5),
            hoverinfo='text',
            text=df_res['Full Language Name'].fillna(df_res['LANGUAGE']),
            name='Languages'
        ))

        for i in range(6):
            cluster_points_res = df_res[df_res['cluster'] == i][['log_unlabelled', 'log_labelled']].values
            if len(cluster_points_res) > 2:
                try:
                    hull_res = ConvexHull(cluster_points_res)
                    hull_points_res = cluster_points_res[hull_res.vertices]
                    hull_points_res = np.vstack([hull_points_res, hull_points_res[0]])
                    tck_res, u_res = splprep([hull_points_res[:,0], hull_points_res[:,1]], s=0, per=True)
                    u_new_res = np.linspace(u_res.min(), u_res.max(), 1000)
                    x_new_res, y_new_res = splev(u_new_res, tck_res, der=0)
                    fig_cluster_res.add_trace(go.Scatter(
                        x=x_new_res,
                        y=y_new_res,
                        mode='lines',
                        line=dict(color=cluster_colors_res[i], dash='dash'),
                        fill='toself',
                        fillcolor=f'rgba({int(cluster_colors_res[i][1:3], 16)}, {int(cluster_colors_res[i][3:5], 16)}, {int(cluster_colors_res[i][5:7], 16)}, 0.3)',
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='none'
                    ))
                except Exception:
                    pass

        fig_cluster_res.add_trace(go.Scatter(
            x=cluster_agg_res['log_unlabelled'],
            y=cluster_agg_res['log_labelled'],
            mode='markers',
            marker=dict(
                size=cluster_agg_res['language_count'],
                sizemode='area',
                sizemin=4,
                sizeref=max(cluster_agg_res['language_count'])/(60.**2),
                color=[cluster_colors_res[c] for c in cluster_agg_res['cluster']],
                showscale=False
            ),
            hoverinfo='text',
            text=[f"Cluster {i}<br>Languages: {lc}<br>Speakers: {format_speakers(ts)}" for i, lc, ts in zip(cluster_agg_res.index, cluster_agg_res['language_count'], cluster_agg_res['total_speakers'])],
            name='Clusters'
        ))

        fig_cluster_res.update_layout(
            title='Language Resource Distribution with Clustering',
            xaxis_title='Log of Unlabelled Resource Count',
            yaxis_title='Log of Labelled Resource Count',
            showlegend=False,
            width=800,
            height=600
        )
        st.plotly_chart(fig_cluster_res, use_container_width=True)
        st.header("Data")
        st.dataframe(df_res)

    except FileNotFoundError:
        st.error('Error: `resource_distribution_with_speakers.csv` not found.')
    st.header("Statistics on Languages per Class")
    try:
        df_taxonomy = pd.read_csv('taxonomy_original_fullnames.csv', header=0)
        st.dataframe(df_taxonomy['Cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'Cluster': 'Number of Languages'}))
    except FileNotFoundError:
        st.error("Error: `taxonomy_original_fullnames.csv` not found.")


with tab2:
    st.header("Our taxonomy (all 7500 languages)")

    try:
        df_all = pd.read_csv('resource_distribution_with_speakers_and_all_languages.csv', delimiter=';')
        df_all['numberOfSpeakers'] = pd.to_numeric(df_all['numberOfSpeakers'], errors='coerce').fillna(0)

        try:
            taxonomy_df_all = pd.read_csv('taxonomy_original_tags_fullnames.csv', header=None)
            taxonomy_df_all.columns = ['Tag', 'Full Language Name', 'Cluster']
            df_all = pd.merge(df_all, taxonomy_df_all[['Full Language Name']], left_on='LANGUAGE', right_on='Full Language Name', how='left')
            df_all['Full Language Name'] = df_all['Full Language Name'].fillna(df_all['LANGUAGE'])
        except FileNotFoundError:
            st.warning('Warning: `taxonomy_original_tags_fullnames.csv` not found. Full language names will not be available.')
            df_all['Full Language Name'] = df_all['LANGUAGE']

        df_all['log_unlabelled'] = np.log10(df_all['UNLABELLED RESOURCE COUNT'].replace(0, 1))
        df_all['log_labelled'] = np.log10(df_all['LABELLED RESOURCE COUNT'].replace(0, 1))

        X_all = df_all[['log_unlabelled', 'log_labelled']]
        kmeans_all = KMeans(n_clusters=6, random_state=42, n_init=10)
        df_all['cluster'] = kmeans_all.fit_predict(X_all)

        centers_all = kmeans_all.cluster_centers_
        center_sums_all = centers_all.sum(axis=1)
        sorted_indices_all = np.argsort(center_sums_all)
        label_mapping_all = {original_label: new_label for new_label, original_label in enumerate(sorted_indices_all)}
        df_all['cluster'] = df_all['cluster'].map(label_mapping_all)
        sorted_centers_all = centers_all[sorted_indices_all]

        cluster_agg_all = df_all.groupby('cluster').agg(
            language_count=('LANGUAGE', 'size'),
            total_speakers=('numberOfSpeakers', 'sum')
        ).reset_index()

        cluster_agg_all['log_unlabelled'] = [center[0] for center in sorted_centers_all]
        cluster_agg_all['log_labelled'] = [center[1] for center in sorted_centers_all]

        cluster_colors_all = ['#6A5ACD', '#483D8B', '#6495ED', '#90EE90', '#FFD700', '#FFA07A', '#CD5C5C']

        fig_cluster_all = go.Figure()
        fig_cluster_all.add_trace(go.Scatter(
            x=df_all['log_unlabelled'],
            y=df_all['log_labelled'],
            mode='markers',
            marker=dict(color='lightgrey', size=5),
            hoverinfo='text',
            text=df_all['Full Language Name'].fillna(df_all['LANGUAGE']),
            name='Languages'
        ))

        for i in range(6):
            cluster_points_all = df_all[df_all['cluster'] == i][['log_unlabelled', 'log_labelled']].values
            if len(cluster_points_all) > 2:
                try:
                    hull_all = ConvexHull(cluster_points_all)
                    hull_points_all = cluster_points_all[hull_all.vertices]
                    hull_points_all = np.vstack([hull_points_all, hull_points_all[0]])
                    tck_all, u_all = splprep([hull_points_all[:,0], hull_points_all[:,1]], s=0, per=True)
                    u_new_all = np.linspace(u_all.min(), u_all.max(), 1000)
                    x_new_all, y_new_all = splev(u_new_all, tck_all, der=0)
                    fig_cluster_all.add_trace(go.Scatter(
                        x=x_new_all,
                        y=y_new_all,
                        mode='lines',
                        line=dict(color=cluster_colors_all[i], dash='dash'),
                        fill='toself',
                        fillcolor=f'rgba({int(cluster_colors_all[i][1:3], 16)}, {int(cluster_colors_all[i][3:5], 16)}, {int(cluster_colors_all[i][5:7], 16)}, 0.3)',
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='none'
                    ))
                except Exception:
                    pass

        fig_cluster_all.add_trace(go.Scatter(
            x=cluster_agg_all['log_unlabelled'],
            y=cluster_agg_all['log_labelled'],
            mode='markers',
            marker=dict(
                size=cluster_agg_all['language_count'],
                sizemode='area',
                sizemin=4,
                sizeref=max(cluster_agg_all['language_count'])/(60.**2),
                color=[cluster_colors_all[c] for c in cluster_agg_all['cluster']],
                showscale=False
            ),
            hoverinfo='text',
            text=[f"Cluster {i}<br>Languages: {lc}<br>Speakers: {format_speakers(ts)}" for i, lc, ts in zip(cluster_agg_all.index, cluster_agg_all['language_count'], cluster_agg_all['total_speakers'])],
            name='Clusters'
        ))

        fig_cluster_all.update_layout(
            title='Language Resource Distribution with Clustering (All Languages)',
            xaxis_title='Log of Unlabelled Resource Count',
            yaxis_title='Log of Labelled Resource Count',
            showlegend=False,
            width=800,
            height=600
        )
        st.plotly_chart(fig_cluster_all, use_container_width=True)
        st.header("Data")
        st.dataframe(df_all)

    except FileNotFoundError:
        st.error('Error: `resource_distribution_with_speakers_and_all_languages.csv` not found.')
    st.header("Statistics on Languages per Class")
    try:
        df_taxonomy = pd.read_csv('taxonomy_all7500_fullnames.csv', header=None, names=['language', 'cluster'])
        st.dataframe(df_taxonomy['cluster'].value_counts().reset_index().rename(columns={'index': 'Cluster', 'cluster': 'Number of Languages'}))
    except FileNotFoundError:
        st.error("Error: `taxonomy_all7500_fullnames.csv` not found.")


with tab3:
    st.header("Taxonomy Comparison (Full Names)")

    # --- Comparison 1 ---
    @st.cache_data
    def load_comparison_1_data():
        try:
            df1 = pd.read_csv('taxonomy_original_fullnames.csv')
            df1.columns = ['language', 'cluster_1']
            df1['language'] = df1['language'].str.lower()
        except FileNotFoundError:
            st.error('Error: `taxonomy_original_fullnames.csv` not found.')
            return None, None
        
        try:
            with open('lang2tax.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                try:
                    lang, cluster = line.strip().rsplit(',', 1)
                    data.append({'language': lang.lower(), 'cluster_2': int(cluster)})
                except ValueError:
                    pass
            df2 = pd.DataFrame(data)
        except FileNotFoundError:
            st.error('Error: `lang2tax.txt` not found.')
            return None, None
            
        return df1, df2

    df1_comp1, df2_comp1 = load_comparison_1_data()
    display_comparison("1) Our Taxonomy vs. Joshi et al. (Full Language Names)", df1_comp1, df2_comp1, "comp1", show_filtered_graphs=True)

    # --- Comparison 2 ---
    @st.cache_data
    def load_comparison_2_data():
        try:
            df1 = pd.read_csv('taxonomy_all7500_fullnames.csv', header=None)
            df1.columns = ['language', 'cluster_1']
            df1['language'] = df1['language'].str.lower()
        except FileNotFoundError:
            st.error('Error: `taxonomy_all7500_fullnames.csv` not found.')
            return None, None

        try:
            with open('lang2tax.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                try:
                    lang, cluster = line.strip().rsplit(',', 1)
                    data.append({'language': lang.lower(), 'cluster_2': int(cluster)})
                except ValueError:
                    pass
            df2 = pd.DataFrame(data)
        except FileNotFoundError:
            st.error('Error: `lang2tax.txt` not found.')
            return None, None
            
        return df1, df2

    df1_comp2, df2_comp2 = load_comparison_2_data()
    display_comparison("2) Our Taxonomy vs. Joshi et al. (All 7500 Languages)", df1_comp2, df2_comp2, "comp2", show_filtered_graphs=False)
