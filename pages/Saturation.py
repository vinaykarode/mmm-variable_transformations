import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------- SATURATION CURVE FUNCTIONS ------------------------

# Hill function
def threshold_hill_function(x, alpha, gamma, threshold=None):
    """
    Compute the value of a Hill function with a threshold for activation.
    The threshold is added for visualisation purposes,
    it makes the graphs display a better S-shape.

    Parameters:
        x (float or array-like): Input variable(s).
        alpha (float): Controls the shape of the curve.
        gamma (float): Controls inflection point of saturation curve.
        threshold (float): Minimum amount of spend before response starts.

    Returns:
        float or array-like: Values of the modified Hill function for the given inputs.
    """
    if threshold:
        # Apply threshold condition
        y = np.where(x > threshold, (x ** alpha) / ((x ** alpha) + (gamma ** alpha)), 0)
    else:
        y = (x ** alpha) / ((x ** alpha) + (gamma ** alpha))
    return y

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

# Logistic function
def logistic_function(x, lam):
    """
    Compute the value of a logistic function for saturation.

    Parameters:
        x (float or array-like): Input variable(s).
        lam (float): Growth rate or steepness of the curve.

    Returns:
        float or array-like: Values of the modified logistic function for the given inputs.
    """
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

# Custom tanh saturation
def tanh_saturation(x, b=0.5, c=0.5):
    """
    Tanh saturation transformation.
    Credit to PyMC-Marketing: https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py

    Parameters:
        x (array-like): Input variable(s).
        b (float): Scales the output. Must be non-negative.
        c (float): Affects the steepness of the curve. Must be non-zero.

    Returns:
        array-like: Transformed values using the tanh saturation formula.
    """
    return b * np.tanh(x / (b * c))


# -------------------------- TOP OF PAGE INFORMATION -------------------------

# Set browser / tab config
st.set_page_config(
    page_title="MMM App - Saturation Curves",
    page_icon="🧊",
)

# Give some context for what the page displays
st.title('Saturation Curves')
st.markdown("This page demonstrates the forms and shapes of saturation curves for MMM.\
             These curves try to model the relationship between weekly marketing \
            spends for a given channel (holding other channels constant) and \
             the conversions that result from that spend.\
            \n It doesn't need to be conversions, it could be sales or customers acquired\
             - whatever target metric you are interested in.\
            ")

st.markdown("**Reminder:** \n \
- Certain saturation functions have **_:red[concave or convex shapes]_**  \n\
- Certain saturation functions have **_:red[S-shapes]_**")

# -------------------------- SATURATION PLOTS -------------------------

# Generate simulated marketing data
np.random.seed(42)
num_points = 500
media_spending = np.linspace(0, 1000, num_points) # x-axis
# Scaling factor for nicer plotting


# Generate simulated datasets with noise
dummy_root = root_function(media_spending, alpha = 0.3) + np.random.normal(0, 0.3, num_points)
dummy_hill = threshold_hill_function(media_spending, alpha = 8, gamma = 400, threshold=200) + np.random.normal(0, 0.05, num_points)
dummy_logistic = logistic_function(media_spending, lam = 0.01) + + np.random.normal(0, 0.1, num_points)
dummy_tanh = tanh_saturation(media_spending, b = 10, c = 20) + + np.random.normal(0, 0.75, num_points)

# Create tabs for Root and Hill plots
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Root", "Hill", "Logistic", "Tanh", "Michaelis-Menten"])

# -------------------------- ROOT CURVE -------------------------
with tab1:
    st.markdown("___The Root curve takes the form:___")
    st.latex(r'''
        x_t^{\textrm{transf}} = x_t^\alpha
        ''')
    st.divider()
    # User inputs
    st.subheader('User Inputs')
    st.markdown('**Try to fit a saturation curve to the generated data!**')

    # User input for Root Curve
    root_alpha = st.slider('**:blue[Root Curve $\\alpha$]:**', 0.0, 1.0, 0.45, key='root_alpha')
    # Calculate the user created response curve
    user_root = root_function(media_spending, alpha=root_alpha)


    # Tidy the simulated dataset for plotting
    plot_data = pd.DataFrame({'Media Spending':np.round(media_spending), 
                            'Conversions':dummy_root})
    # Drop rows with negative conversions, generated by the noise
    plot_data = plot_data[plot_data.Conversions >= 0]
    
    # Plot
    fig_root = go.Figure()
    # Plot weekly spend and response data, every 5th to make the plot less crowded
    fig_root.add_trace(go.Scatter(x = plot_data['Media Spending'][::5],
                            y = plot_data['Conversions'][::5],
                            mode = 'markers',
                            name = 'Weekly Data',
                            marker = dict(color='#AB63FA')))
    # Plot user-defined curve to match that data
    fig_root.add_trace(go.Scatter(x=media_spending,
                                  y=user_root,
                                  mode='lines',
                                  name='Saturation Curve',
                                  line=dict(color='blue', dash='solid')))
    
    fig_root.update_layout(title_text="Root Saturation Curve", 
                           xaxis_title='Media Spend',
                           yaxis_title='Conversions',
                           height=500, width=1000)
                           
    st.plotly_chart(fig_root, use_container_width=True)

# -------------------------- HILL CURVE -------------------------
with tab2:
    st.markdown("___The Hill function takes the form:___")
    st.latex(r'''
        x_t^{\textrm{transf}} = \frac{x_t^\alpha}{x_t^\alpha + \gamma^\alpha}
        ''')
    st.divider()
    # User inputs
    st.subheader('User Inputs')
    st.markdown('**Try to fit a saturation curve to the generated data!**')

    # User input for Hill Curve
    hill_alpha = st.slider(':red[Hill Curve $\\alpha$]:', 0.0, 10.0, 0.45)
    hill_gamma = st.slider(':red[Hill Curve $\\gamma$]:', 1, 1000, 100)
    # Calculate the user created response curve
    user_hill = threshold_hill_function(media_spending, alpha = hill_alpha, gamma = hill_gamma)


    # Tidy the simulated dataset for plotting
    plot_data = pd.DataFrame({'Media Spending':np.round(media_spending), 
                            'Conversions':dummy_hill})
    # Drop rows with negative conversions, generated by the noise
    plot_data = plot_data[plot_data.Conversions >= 0]
    
    # Plot
    fig_root = go.Figure()
    # Plot weekly spend and response data, every 5th to make the plot less crowded
    fig_root.add_trace(go.Scatter(x = plot_data['Media Spending'][::5],
                            y = plot_data['Conversions'][::5],
                            mode = 'markers',
                            name = 'Weekly Data',
                            marker = dict(color='#AB63FA')))
    # Plot user-defined curve to match that data
    fig_root.add_trace(go.Scatter(x=media_spending,
                                  y=user_hill,
                                  mode='lines',
                                  name='Saturation Curve',
                                  line=dict(color='blue', dash='solid')))
    
    fig_root.update_layout(title_text="Hill Saturation Curve", 
                           xaxis_title='Media Spend',
                           yaxis_title='Conversions',
                           height=500, width=1000)
                           
    st.plotly_chart(fig_root, use_container_width=True)


# -------------------------- LOGISTIC CURVE -------------------------
with tab3:
    st.markdown("___The Logistic function takes the form:___")
    st.latex(r'''
        x_t^{\textrm{transf}} = \frac{1 - e^{-\lambda x}}{1 + e^{-\lambda x}}
        ''')
    st.caption(":link: Credit to [pymc-marketing](https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py) for this code")
    st.divider()
    # User inputs
    st.subheader('User Inputs')
    st.markdown('**Try to fit a saturation curve to the generated data!**')

    # User input for Modified Logistic Curve
    logistic_lam = st.slider(':green[Logistic Curve $\\lambda$ (scaled value):]', 0, 1000, 500, step=1, key='logistic_lam')  
    logistic_lam = logistic_lam / 10000

    # Calculate the user created response curve
    user_logistic = logistic_function(media_spending, lam=logistic_lam)


    # Tidy the simulated dataset for plotting
    plot_data = pd.DataFrame({'Media Spending':np.round(media_spending), 
                            'Conversions':dummy_logistic})
    # Drop rows with negative conversions, generated by the noise
    plot_data = plot_data[plot_data.Conversions >= 0]
    
    # Plot
    fig_root = go.Figure()
    # Plot weekly spend and response data, every 5th to make the plot less crowded
    fig_root.add_trace(go.Scatter(x = plot_data['Media Spending'][::5],
                            y = plot_data['Conversions'][::5],
                            mode = 'markers',
                            name = 'Weekly Data',
                            marker = dict(color='#AB63FA')))
    # Plot user-defined curve to match that data
    fig_root.add_trace(go.Scatter(x=media_spending,
                                  y=user_logistic,
                                  mode='lines',
                                  name='Saturation Curve',
                                  line=dict(color='blue', dash='solid')))
    
    fig_root.update_layout(title_text="Logistic Saturation Curve", 
                           xaxis_title='Media Spend',
                           yaxis_title='Conversions',
                           height=500, width=1000)
                           
    st.plotly_chart(fig_root, use_container_width=True)


# -------------------------- TANH CURVE -------------------------
with tab4:
    st.markdown("___The Tanh saturation function takes the form:___")
    st.latex(r'''
        x_t^{\textrm{transf}} = b \tanh \left( \frac{x}{bc} \right)
        ''')
    st.caption(":link: Credit to [pymc-marketing](https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py) for this code")
    st.divider()
    # User inputs
    st.subheader('User Inputs')
    st.markdown('**Try to fit a saturation curve to the generated data!**')

    # User input for Tanh Curve
    tanh_b = st.slider(':red[Tanh Curve $\\text{b}$]:', 0, 20, 5)
    tanh_c = st.slider(':red[Tanh Curve $\\text{c}$]:', 0, 100, 50)

    # Calculate the user created response curve
    user_tanh = tanh_saturation(media_spending, b=tanh_b, c=tanh_c)


    # Tidy the simulated dataset for plotting
    plot_data = pd.DataFrame({'Media Spending':np.round(media_spending), 
                            'Conversions':dummy_tanh})
    # Drop rows with negative conversions, generated by the noise
    plot_data = plot_data[plot_data.Conversions >= 0]
    
    # Plot
    fig_root = go.Figure()
    # Plot weekly spend and response data, every 5th to make the plot less crowded
    fig_root.add_trace(go.Scatter(x = plot_data['Media Spending'][::5],
                            y = plot_data['Conversions'][::5],
                            mode = 'markers',
                            name = 'Weekly Data',
                            marker = dict(color='#AB63FA')))
    # Plot user-defined curve to match that data
    fig_root.add_trace(go.Scatter(x=media_spending,
                                  y=user_tanh,
                                  mode='lines',
                                  name='Saturation Curve',
                                  line=dict(color='blue', dash='solid')))
    
    fig_root.update_layout(title_text="Tanh Saturation Curve", 
                           xaxis_title='Media Spend',
                           yaxis_title='Conversions',
                           height=500, width=1000)
                           
    st.plotly_chart(fig_root, use_container_width=True)

# # Select values to generate response curves with
# root_alpha = st.slider(':blue[Root Curve Alpha] :large_blue_square:', 0.0, 1.0, 0.45)
# hill_alpha = st.slider(':red[Hill Curve Alpha] :large_red_square:', 0.0, 10.0, 0.45)
# hill_gamma = st.slider(':red[Hill Curve Gamma] :large_red_square:', 1, 100, 1)

# # Let user select what kind of spend / response data to plot
# data_option = st.radio(
#     '**Please select a fictional dataset to plot your response curves on top of:**',
#     ('Dataset A', 'Dataset B'))

# if data_option == 'Dataset A':
#     response_data = dummy_root
# elif data_option == 'Dataset B':
#     response_data = dummy_hill

# # Plot the spend / response data
# plot_data = pd.DataFrame({'Media Spending':np.round(media_spending), 
#                           'Conversions':response_data * scaling_factor})
# # Drop rows with negative conversions, generated by the noise
# plot_data = plot_data[plot_data.Conversions >= 0]

# # Calculate the user created response curves
# user_root = root_function(plot_data["Media Spending"], alpha = root_alpha) * scaling_factor
# user_hill = hill_function(plot_data["Media Spending"], alpha = hill_alpha, gamma = hill_gamma) * scaling_factor

# # Plot simulated spend and conversion data
# fig = go.Figure()
# fig.add_trace(go.Scatter(x = plot_data['Media Spending'][::5],
#                          y = plot_data['Conversions'][::5],
#                          mode = 'markers',
#                          name = 'Simulated Data',
#                          marker = dict(color='#AB63FA')))

# # Plot the user-created response curves
# fig.add_trace(go.Scatter(x=plot_data['Media Spending'],
#                          y=user_root,
#                          mode='lines',
#                          name='Root Curve',
#                          line=dict(color='blue', dash='solid')))

# fig.add_trace(go.Scatter(x=plot_data['Media Spending'],
#                          y=user_hill,
#                          mode='lines',
#                          name='Hill Curve',
#                          line=dict(color='red', dash='solid')))

# # Customise plot
# fig.layout.height = 500
# fig.layout.width = 1000
# fig.update_layout(title_text="Saturation Curves of Marketing Spend vs Conversions", 
#                   xaxis_title='Media Spend',
#                   yaxis_title='Response',
#                 #   legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
#                   title_font = dict(size = 30))
# st.plotly_chart(fig, theme="streamlit", use_container_width=False)