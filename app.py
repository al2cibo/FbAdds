import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Set page to wide mode
st.set_page_config(layout="wide")

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
st.title('Enhanced Advertising Campaign Analysis Dashboard')

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Tier 4 Analysis", "Tier 5 Analysis", "Time Series Analysis", "Insights & Recommendations"])

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
    }), use_container_width=True)

    # Visualizations
    st.subheader('Campaign Visualizations')

    # Scatter plot of Spend vs Conversions
    fig = px.scatter(campaign_data.dropna(subset=['ROAS']), x='Spend', y='RB Conv', size='ROAS', color='ROAS',
                     hover_name=tier_column, log_x=True, size_max=60,
                     labels={'Spend': 'Total Spend ($)', 'RB Conv': 'Total Conversions', 'ROAS': 'ROAS'},
                     title=f'Spend vs Conversions (Size/Color = ROAS) for {tier_column}')
    st.plotly_chart(fig, use_container_width=True)

    # Bar plot of ROAS by Campaign
    top_campaigns = campaign_data.sort_values('ROAS', ascending=False).head(10)
    fig = px.bar(top_campaigns, x=tier_column, y='ROAS', color='Spend',
                 labels={'ROAS': 'Return on Ad Spend', 'Spend': 'Total Spend ($)'},
                 title=f'Top 10 Campaigns by ROAS for {tier_column}')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    corr = campaign_data[['Spend', 'RB Conv', 'RB CPO', 'AOV', 'ROAS', 'CPA', 'Revenue']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    title=f'Correlation Heatmap of Metrics for {tier_column}')
    st.plotly_chart(fig, use_container_width=True)

    return campaign_data, corr

# Tier 4 Analysis
with tab1:
    tier4_data, tier4_corr = campaign_analysis(df, 'Tier 4')

# Tier 5 Analysis
with tab2:
    tier5_data, tier5_corr = campaign_analysis(df, 'Tier 5')

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
    
    # Line plot for all metrics over time
    fig = go.Figure()
    
    for metric in ['Spend', 'RB Conv', 'RB CPO', 'AOV', 'ROAS']:
        fig.add_trace(go.Scatter(x=time_df['Week'], y=time_df[metric], name=metric, visible='legendonly' if metric != 'ROAS' else True))
    
    fig.update_layout(title_text='Metrics Over Time', height=600)
    fig.update_xaxes(title_text='Week')
    fig.update_yaxes(title_text='Value')
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly performance table
    st.subheader('Weekly Performance')
    st.dataframe(time_df.style.format({
        'Spend': '${:,.2f}',
        'RB Conv': '{:,.2f}',
        'RB CPO': '${:,.2f}',
        'AOV': '${:,.2f}',
        'ROAS': '{:,.2f}'
    }), use_container_width=True)


    roas_series = time_df.set_index('Week')['ROAS']
    # ARIMA forecast
    st.subheader('ROAS Forecast')
    model = ARIMA(roas_series, order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=4)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roas_series.index, y=roas_series.values, name='Historical ROAS'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecasted ROAS', line=dict(dash='dash')))
    fig.update_layout(title='ROAS Forecast for Next 4 Weeks', xaxis_title='Week', yaxis_title='ROAS')
    st.plotly_chart(fig, use_container_width=True)

# Insights and Recommendations
with tab4:
    st.header('Key Insights and Recommendations')

    # Tier 4 Insights
    st.subheader('Tier 4 Campaign Insights')
    top_tier4 = tier4_data.loc[tier4_data['ROAS'].idxmax()]
    worst_tier4 = tier4_data.loc[tier4_data['ROAS'].idxmin()]
    highest_spend_tier4 = tier4_data.loc[tier4_data['Spend'].idxmax()]
    
    st.write(f"""
    1. Best Performing Tier 4 Campaign: '{top_tier4['Tier 4']}' 
       - ROAS: {top_tier4['ROAS']:.2f}
       - Revenue: ${top_tier4['Revenue']:,.2f}
       - Spend: ${top_tier4['Spend']:,.2f}
    
    2. Underperforming Tier 4 Campaign: '{worst_tier4['Tier 4']}'
       - ROAS: {worst_tier4['ROAS']:.2f}
       - Revenue: ${worst_tier4['Revenue']:,.2f}
       - Spend: ${worst_tier4['Spend']:,.2f}
    
    3. Highest Spend Tier 4 Campaign: '{highest_spend_tier4['Tier 4']}'
       - Spend: ${highest_spend_tier4['Spend']:,.2f}
       - ROAS: {highest_spend_tier4['ROAS']:.2f}
       - Revenue: ${highest_spend_tier4['Revenue']:,.2f}
    
    4. Tier 4 Metric Correlations:
       - Spend vs. Conversions: {tier4_corr.loc['Spend', 'RB Conv']:.2f}
       - ROAS vs. Spend: {tier4_corr.loc['ROAS', 'Spend']:.2f}
    """)

    # Tier 5 Insights
    st.subheader('Tier 5 Campaign Insights')
    top_tier5 = tier5_data.loc[tier5_data['ROAS'].idxmax()]
    worst_tier5 = tier5_data.loc[tier5_data['ROAS'].idxmin()]
    highest_spend_tier5 = tier5_data.loc[tier5_data['Spend'].idxmax()]
    
    st.write(f"""
    1. Best Performing Tier 5 Campaign: '{top_tier5['Tier 5']}' 
       - ROAS: {top_tier5['ROAS']:.2f}
       - Revenue: ${top_tier5['Revenue']:,.2f}
       - Spend: ${top_tier5['Spend']:,.2f}
    
    2. Underperforming Tier 5 Campaign: '{worst_tier5['Tier 5']}'
       - ROAS: {worst_tier5['ROAS']:.2f}
       - Revenue: ${worst_tier5['Revenue']:,.2f}
       - Spend: ${worst_tier5['Spend']:,.2f}
    
    3. Highest Spend Tier 5 Campaign: '{highest_spend_tier5['Tier 5']}'
       - Spend: ${highest_spend_tier5['Spend']:,.2f}
       - ROAS: {highest_spend_tier5['ROAS']:.2f}
       - Revenue: ${highest_spend_tier5['Revenue']:,.2f}
    
    4. Tier 5 Metric Correlations:
       - Spend vs. Conversions: {tier5_corr.loc['Spend', 'RB Conv']:.2f}
       - ROAS vs. Spend: {tier5_corr.loc['ROAS', 'Spend']:.2f}
    """)

    # Time Series Insights
    st.subheader('Time Series Insights')
    best_week = time_df.loc[time_df['ROAS'].idxmax()]
    worst_week = time_df.loc[time_df['ROAS'].idxmin()]
    spend_trend = 'increasing' if time_df['Spend'].iloc[-1] > time_df['Spend'].iloc[0] else 'decreasing'
    roas_trend = 'improving' if time_df['ROAS'].iloc[-1] > time_df['ROAS'].iloc[0] else 'declining'
    
    st.write(f"""
    1. Best Performing Week: {best_week['Week'].date()}
       - ROAS: {best_week['ROAS']:.2f}
       - Spend: ${best_week['Spend']:,.2f}
       - Conversions: {best_week['RB Conv']:.0f}
    
    2. Worst Performing Week: {worst_week['Week'].date()}
       - ROAS: {worst_week['ROAS']:.2f}
       - Spend: ${worst_week['Spend']:,.2f}
       - Conversions: {worst_week['RB Conv']:.0f}
    
    3. Overall Trends:
       - Spend is {spend_trend} over time
       - ROAS is {roas_trend} over the analyzed period

           
    4. Forecast: The ARIMA model predicts a {'rising' if forecast.values[-1] > roas_series.iloc[-1] else 'falling'} trend in ROAS for the next 4 weeks.
    """)

    st.subheader('Recommendations')
    st.write("""
    Based on the comprehensive data analysis, here are key recommendations:

    1. Campaign Optimization:
       - Focus resources on top-performing campaigns in both Tier 4 and Tier 5.
       - Review and optimize or pause the lowest-performing campaigns, especially those with high spend and low ROAS.

    2. Budget Allocation:
       - Redistribute budget from underperforming campaigns to those showing high ROAS and potential for scaling.
       - Consider increasing overall spend if there's a positive correlation between spend and ROAS.

    3. Performance Metrics:
       - Monitor the relationship between CPA and ROAS closely. Prioritize campaigns with low CPA and high ROAS for increased investment.
       - Pay attention to AOV and its impact on overall revenue. Campaigns driving higher AOV might be more valuable even with slightly lower conversion rates.

    4. Seasonal Strategy:
       - Align campaign efforts with identified peak performance periods from the time series analysis.
       - Prepare and allocate resources for historically high-performing weeks or seasons.

    5. Forecasting and Trend Analysis:
       - Use the ARIMA forecast to guide short-term strategy. Adjust budget and campaign focus based on the predicted ROAS trend.
       - Regularly update the forecast model with new data to improve accuracy.

    6. Testing and Innovation:
       - Implement A/B testing for ad creatives, targeting options, and bidding strategies, especially for mid-performing campaigns with potential for improvement.
       - Explore new ad formats or platforms that align with high-performing campaign characteristics.

    7. Conversion Rate Optimization:
       - For campaigns with high spend but lower conversion rates, review and optimize landing pages and user experience to improve conversion rates.
         - Implement retargeting strategies to capture missed conversions and improve overall campaign performance.
             """)
    
