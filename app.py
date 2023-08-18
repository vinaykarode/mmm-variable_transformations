import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import weibull_min
from sklearn.preprocessing import MinMaxScaler
# ------------------- VARIABLE TRANSFORMATION FUNCTIONS ------------------------

def geometric_adstock(impact, decay_factor, periods):
    """
    Calculate the geometric adstock effect.

    Parameters:
        impact (float): Initial advertising impact.
        decay_factor (float): Decay factor between 0 and 1.
        periods (int): Number of periods.

    Returns:
        list: List of adstock values for each period.
    """
    adstock_values = [impact]
    
    for _ in range(1, periods):
        impact *= decay_factor
        adstock_values.append(impact)
    
    return adstock_values

def weibull_pdf_adstock_decay(impact, shape, scale, periods):
    """
    Calculate Weibull PDF adstock decay for media mix modeling.

    Parameters:
        impact (float): Initial advertising impact.
        shape (float): Shape parameter of the Weibull distribution.
        scale (float): Scale parameter of the Weibull distribution.
        periods (int): Number of periods.

    Returns:
        list: List of adstock-transformed values for each period.
    """
    # List to store decay values
    adstock_decay_values = []

    # Transform the scale parameter according to percentile of time period
    transformed_scale = np.percentile(range(1, periods), scale * 100) 
    
    # Create a Weibull distribution object with the transformed scale parameter
    weibull_dist = weibull_min(shape, scale=transformed_scale)

    # Calculate adstock decay values for each period using the Weibull PDF
    for t in range(1, periods+1):
        adstock_decay = impact * weibull_dist.pdf(t)
        adstock_decay_values.append(adstock_decay)

    # Normalize adstock decay values for easier plotting
    reshaped_decay = np.array(adstock_decay_values).reshape(-1, 1)
    normalised_adstock_decay = MinMaxScaler().fit_transform(reshaped_decay).flatten()
    
    return normalised_adstock_decay

# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Give some context for what the page displays
st.title('Adstock Transformations')
st.markdown("This web app demonstrates the effect of various adstock \
            transformations on a variable.  \nFor these examples, let's imagine \
            that we have _some variable that represents a quantity of a particlar_ \
            _advertising channel_.  \nFor example, this could be the number of impressions\
            we get from Facebook.  \nSo, at the start of our example (_Week 1_), \
            we would have **100 impressions from Facebook**. \
            \n\n**_:violet[We will use this starting value of 100 for all of our adstock examples]_**. \
            ")

# Starting value for adstock
initial_impact = 100

# Separate the adstock transformations into 3 tabs
tab1, tab2, tab3 = st.tabs(["Geometric", "Weibull CDF", "Weibull PDF"])

# -------------------------- GEOMETRIC ADSTOCK DISPLAY -------------------------
with tab1:
    st.header('Geometric Adstock Transformation')

    # User inputs
    st.subheader('User Inputs')
    num_periods = st.slider('Geometric - Number of weeks after impressions first received :alarm_clock:', 1, 100, 20)
    # Let user choose 3 decay rates to compare simultaneously
    decay_rate_1 = st.slider(':blue[Beta 1] :large_blue_square:', 0.0, 1.0, 0.3)
    decay_rate_2 = st.slider(':red[Beta 2] :large_red_square:', 0.0, 1.0, 0.6)
    decay_rate_3 = st.slider(':green[Beta 3] :large_green_square:', 0.0, 1.0, 0.9)

    # Create a list of decay rates
    decay_rates = [decay_rate_1, decay_rate_2, decay_rate_3]

    # Create df to store each adstock in
    all_adstocks = pd.DataFrame()
    # Iterate through decay rates and generate df of values to plot
    for i, beta in enumerate(decay_rates):
        # Get geometric adstock values, decayed over time
        adstock_df = pd.DataFrame({"Week": range(1, (num_periods + 1)),
                                ## Calculate adstock values
                                    "Adstock": geometric_adstock(initial_impact, beta, num_periods),
                                    ## Format adstock labels for neater plotting
                                "Adstock Labels": [f'{x:,.0f}' for x in geometric_adstock(initial_impact, beta, num_periods)], 
                                    ## Create column to label each adstock
                                    "Beta": f"Beta {i + 1}"})

        all_adstocks = pd.concat([all_adstocks, adstock_df])

    # Plot adstock values
    # Annotate the plot if user wants it
    st.markdown('**Would you like to show the adstock values directly on the plot?**')
    annotate = st.checkbox('Yes please! :pray:')
    if annotate:
        fig = px.line(all_adstocks, x = 'Week',
                y = 'Adstock', text = 'Adstock Labels',
                markers=True, color = "Beta",
                # Replaces default color mapping by value
                color_discrete_map={"Beta 1": "#636EFA",
                                        "Beta 2": "#EF553B",
                                        "Beta 3": "#00CC96"})
        fig.update_traces(textposition="bottom left")
    else:
        fig = px.line(all_adstocks, x = 'Week',
                y = 'Adstock',
                markers=True, color = "Beta",
                # Replaces default color mapping by value
                color_discrete_map={"Beta 1": "#636EFA",
                                        "Beta 2": "#EF553B",
                                        "Beta 3": "#00CC96"})
    # Format plot
    fig.layout.height = 600
    fig.layout.width = 1000
    fig.update_layout(title_text="Geometric Adstock Decayed Over Weeks", 
                    title_font = dict(size = 30))
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)


# -------------------------- WEIBULL CDF ADSTOCK DISPLAY -------------------------
with tab2:
    st.header('Weibull CDF Adstock Transformation')
    # User inputs
    st.subheader('User Inputs')

# -------------------------- WEIBULL PDF ADSTOCK DISPLAY -------------------------
with tab3:
    st.header('Weibull PDF Adstock Transformation')

    # User inputs
    st.subheader('User Inputs')
    num_periods_3 = st.slider('Weibull PDF - Number of weeks after impressions first received :alarm_clock:', 1, 100, 20)
    # Let user choose shape and scale parameters to compare two Weibull PDF decay curves simultaneously
    # Params for Line A
    shape_parameter_A = st.slider(':blue[Line A] :large_blue_square:', 0.0, 10, 2)
    scale_parameter_A = st.slider(':triangular_ruler::blue[Line A] :large_blue_square:', 0.0, 1, 0.5)
    # Params for Line B
    shape_parameter_B = st.slider(':red[Line B] :large_red_square:', 0.0, 10, 0.5)
    scale_parameter_B = st.slider(':triangular_ruler::red[Line B] :large_red_square:', 0.0, 1, 0.01)

    # Calculate weibull pdf adstock values, decayed over time for both sets of params
    adstock_series_A = weibull_pdf_adstock_decay(initial_impact, shape_parameter_A,
                                                  scale_parameter_A, num_periods_3)
    adstock_series_B = weibull_pdf_adstock_decay(initial_impact, shape_parameter_B,
                                                  scale_parameter_B, num_periods_3)

    # Create dfs of both sets of adstock values, to plot with
    adstock_df_A = pd.DataFrame({"Week": range(1, (num_periods_3 + 1)),
                                "Adstock": adstock_series_A,
                                "Line": "Line A"})
    adstock_df_B = pd.DataFrame({"Week": range(1, (num_periods_3 + 1)),
                                "Adstock": adstock_series_B,
                                "Line": "Line B"})
    # Create plotting df
    weibull_pdf_df = pd.concat([adstock_df_A, adstock_df_B])
    # Multiply by 100 to get back to scale of initial impact (100 FB impressions)
    weibull_pdf_df.Adstock = weibull_pdf_df.Adstock * 100
    # Format adstock labels for neater plotting
    weibull_pdf_df["Adstock Labels"] = weibull_pdf_df.Adstock.map('{:,.0f}'.format)

    # Plot adstock values
    # Annotate the plot if user wants it
    st.markdown('**Would you like to show the adstock values directly on the plot?**')
    annotate = st.checkbox('Yes please! :pray:')
    if annotate:
        fig = px.line(weibull_pdf_df, x = 'Week',
                y = 'Adstock', text = 'Adstock Labels',
                markers=True, color = "Line",
                # Replaces default color mapping by value
                color_discrete_map={"Line A": "#636EFA",
                                        "Line B": "#EF553B"})
        fig.update_traces(textposition="bottom left")
    else:
        fig = px.line(weibull_pdf_df, x = 'Week',
                y = 'Adstock',
                markers=True, color = "Line",
                # Replaces default color mapping by value
                color_discrete_map={"Line A": "#636EFA",
                                        "Line B": "#EF553B"})
    # Format plot
    fig.layout.height = 600
    fig.layout.width = 1000
    fig.update_layout(title_text="Weibull PDF Adstock Decayed Over Weeks", 
                    title_font = dict(size = 30))
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)