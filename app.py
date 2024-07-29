import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    df['Week'] = pd.to_datetime(df['Week'].str.split('-').str[0])
    numeric_columns = ['ROAS', 'Spend', 'RB Conv', 'RB CPO', 'AOV']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    return df

df = load_data()

# Streamlit app
st.title('Advertising Campaign Analysis Dashboard')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Tier 4 Analysis", "Tier 5 Analysis", "Time Series Analysis"])

# Function to create campaign analysis
def campaign_analysis(df, tier_column):
    # Aggregate data by campaign
    campaign_data = df.groupby(tier_column).agg({
        'Spend': 'sum',
        'RB Conv': 'sum',
        'RB CPO': 'mean',
        'AOV': 'mean',
        'ROAS': 'mean'
    }).reset_index()
    
    # Calculate additional metrics
    campaign_data['CPA'] = campaign_data['Spend'] / campaign_data['RB Conv']
    campaign_data['Revenue'] = campaign_data['Spend'] * campaign_data['ROAS']
    
    # Display campaign performance table
    st.subheader(f'{tier_column} Campaign Performance')
    st.dataframe(campaign_data.sort_values('ROAS', ascending=False).style.format({
        'Spend': '${:,.2f}',
        'RB Conv': '{:,.2f}',
        'RB CPO': '${:,.2f}',
        'AOV': '${:,.2f}',
        'ROAS': '{:,.2f}',
        'CPA': '${:,.2f}',
        'Revenue': '${:,.2f}'
    }),use_container_width=True)

    # Visualizations
    st.subheader('Campaign Visualizations')

    
    col1, col2 = st.columns(2)
    with col1:

        # Scatter plot of Spend vs Conversions
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=campaign_data, x='Spend', y='RB Conv', size='ROAS', hue='ROAS', palette='viridis', ax=ax)
        plt.title(f'Spend vs Conversions (Size/Color = ROAS) for {tier_column}')
        st.pyplot(fig)


    with col2:

        # Bar plot of ROAS by Campaign
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=campaign_data.sort_values('ROAS', ascending=False).head(10), x=tier_column, y='ROAS', ax=ax)
        plt.title(f'Top 10 Campaigns by ROAS for {tier_column}')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader('Metric Correlations')
    corr = campaign_data[['Spend', 'RB Conv', 'RB CPO', 'AOV', 'ROAS', 'CPA', 'Revenue']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title(f'Correlation Heatmap of Metrics for {tier_column}')
    st.pyplot(fig)

# Tier 4 Analysis
with tab1:
    campaign_analysis(df, 'Tier 4')

# Tier 5 Analysis
with tab2:
    campaign_analysis(df, 'Tier 5')

# Time Series Analysis
with tab3:
    st.subheader('Time Series Analysis')
    
    # Aggregate data by week
    time_df = df.groupby('Week').agg({
        'Spend': 'sum',
        'RB Conv': 'sum',
        'RB CPO': 'mean',
        'AOV': 'mean',
        'ROAS': 'mean'
    }).reset_index()
    
    # Line plot for selected metrics over time
    metrics = ['Spend', 'RB Conv', 'RB CPO', 'AOV', 'ROAS']
    selected_metrics = st.multiselect('Select metrics for time series analysis', metrics, default=['Spend', 'ROAS'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for metric in selected_metrics:
        sns.lineplot(data=time_df, x='Week', y=metric, label=metric, ax=ax)
    plt.title('Metrics Over Time')
    plt.legend()
    st.pyplot(fig)
    
    # Weekly performance table
    st.subheader('Weekly Performance')
    st.dataframe(time_df.style.format({
        'Spend': '${:,.2f}',
        'RB Conv': '{:,.2f}',
        'RB CPO': '${:,.2f}',
        'AOV': '${:,.2f}',
        'ROAS': '{:,.2f}'
    }),use_container_width=True)

# Insights and Recommendations
st.header('Insights and Recommendations')
st.write("""
Based on the data analysis, here are some key insights and recommendations:

1. Top-performing campaigns: Identify the campaigns with the highest ROAS in both Tier 4 and Tier 5. Consider allocating more budget to these high-performing campaigns.

2. Spend efficiency: Analyze the relationship between spend and conversions. Look for campaigns that have a good balance of high spend and high conversions.

3. Optimize underperforming campaigns: Identify campaigns with low ROAS and high CPA. Consider adjusting or pausing these campaigns to improve overall performance.

4. Scaling opportunities: Look for campaigns with high ROAS but relatively low spend. These might be opportunities to scale up and increase overall revenue.

5. Time-based trends: Use the time series analysis to identify any seasonal trends or patterns in performance. This can inform future campaign planning and budget allocation.

6. Metric relationships: Use the correlation heatmaps to understand relationships between different metrics. This can help in identifying key drivers of performance.

7. Campaign structure: Compare the performance of Tier 4 vs Tier 5 campaigns. This might provide insights into which level of campaign granularity is more effective.

8. Revenue focus: While ROAS is important, also pay attention to total revenue generated. Some campaigns might have slightly lower ROAS but generate significantly more revenue.

9. Testing and iteration: For underperforming campaigns, consider A/B testing different ad creatives, targeting options, or bidding strategies to improve their performance.

10. Budget allocation: Based on the performance data, reassess your budget allocation across campaigns to maximize overall ROAS and revenue.
""")

# Footer
st.sidebar.markdown('---')
st.sidebar.write('Dashboard created with Streamlit')