import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import weibull_min
from sklearn.preprocessing import MinMaxScaler

# ------------------- SATURATION CURVE FUNCTIONS ------------------------

# Hill function
def hill_function(x, alpha, gamma):
    """
    Compute the value of a Hill function for saturation.

    Parameters:
        x (float or array-like): Input variable(s).
        alpha (float): Controls the shape of the curve.
        gamma (float): Controls inflection point of saturation curve.

    Returns:
        float or array-like: Values of the Hill function for the given inputs.
    """
    return (x ** alpha) / ((x ** alpha) + (gamma ** alpha))

# Root function
def root_function(x, alpha):
    """
    Compute the value of a root function.
    The root function raises the input variable to a power specified by the alpha parameter.

    Parameters:
        x (float or array-like): Input variable(s).
        alpha (float): Exponent controlling the root function.

    Returns:
        float or array-like: Values of the root function for the given inputs.
    """
    return x ** alpha


# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - Saturation Curves",
    page_icon="🧊",
)

# Give some context for what the page displays
st.title('Saturation Curves')
st.markdown("This page demonstrates the forms and shapes of saturation curves for MMM.\
            The saturation curves considered here are either root curves or \
            Hill curves. Both try to model the relationship between marketing \
            spend for a given channel (holding other channels constant) and \
             the conversions that result from that spend.\
            ")


# -------------------------- ROOT AND HILL PLOTS -------------------------

# Generate simulated marketing data
np.random.seed(42)
num_points = 400
media_spending = np.linspace(0, 500, num_points) # x-axis

# User inputs
st.subheader('User Inputs')

# Select values to generate response curves with
root_alpha = st.slider(':blue[Root Curve Alpha] :large_blue_square:', 0.0, 1.0, 0.45)
hill_alpha = st.slider(':red[Hill Curve Alpha] :large_red_square:', 0.0, 10.0, 0.45)
hill_gamma = st.slider(':red[Hill Curve Gamma] :large_red_square:', 1, 100, 1)
# Scaling factor for nicer plotting
scaling_factor = 1000

# Generate simulated datasets with noise
dummy_root = root_function(media_spending, alpha = 0.3) + np.random.normal(0, 0.3, num_points)
dummy_hill = hill_function(media_spending, alpha = 8, gamma = 100) + np.random.normal(0, 0.1, num_points) 

# Let user select what kind of spend / response data to plot
data_option = st.radio(
    '**Please select a fictional dataset to plot your response curves on top of**',
    ('Dataset A', 'Dataset B'))

if data_option == 'Dataset A':
    response_data = dummy_root
elif data_option == 'Dataset B':
    response_data = dummy_hill

# Plot the spend / response data
plot_data = pd.DataFrame({'Media Spending':np.round(media_spending), 
                          'Conversions':response_data * scaling_factor})
# Drop rows with negative conversions, generated by the noise
plot_data = plot_data[plot_data.Conversions >= 0]

# Calculate the user created response curves
user_root = root_function(plot_data["Media Spending"], alpha = root_alpha) * scaling_factor
user_hill = hill_function(plot_data["Media Spending"], alpha = hill_alpha, gamma = hill_gamma) * scaling_factor

# Plot simulated spend and conversion data
fig = go.Figure()
fig.add_trace(go.Scatter(x = plot_data['Media Spending'],
                         y = plot_data['Conversions'],
                         mode = 'markers',
                         name = 'Simulated Data',
                         marker = dict(color='#AB63FA')))

# Plot the user-created response curves
fig.add_trace(go.Scatter(x=plot_data['Media Spending'],
                         y=user_root,
                         mode='lines',
                         name='Root Curve',
                         line=dict(color='blue', dash='solid')))

fig.add_trace(go.Scatter(x=plot_data['Media Spending'],
                         y=user_hill,
                         mode='lines',
                         name='Hill Curve',
                         line=dict(color='red', dash='solid')))

# Customise plot
fig.layout.height = 500
fig.layout.width = 1000
fig.update_layout(title_text="Saturation Curves of Marketing Spend vs Conversions", 
                  xaxis_title='Media Spend',
                  yaxis_title='Response',
                #   legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                  title_font = dict(size = 30))
st.plotly_chart(fig, theme="streamlit", use_container_width=False)