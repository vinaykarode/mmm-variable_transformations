import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# -------------------------- GEOMETRIC ADSTOCK DISPLAY ---------------------------------
st.title('Adstock Transformations')
# Give some context for what the page displays
st.markdown("This page demonstrates the effect of various adstock \
            transformations on a variable.  \nFor these examples, let's imagine \
            that we have _some variable that represents a quantity of a particlar_ \
            _advertising channel_.  \nFor example, this could be the number of impressions\
            we get from Facebook.  \nSo, at the start of our example (_Week 1_), \
            we would have **100 impressions from Facebook**. \
            \n\n**_:violet[We will use this starting value of 100 for all of our adstock examples]_**. \
            ")


st.header('Geometric Adstock Transformation')


# Starting value for adstock
initial_impact = 100
# User inputs
st.subheader('User Inputs')
num_periods = st.slider('Number of Weeks :alarm_clock:', 1, 100, 20)
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
