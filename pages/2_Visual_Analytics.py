import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components
st.set_page_config(layout="wide")



# Load the data
data = pd.read_csv("data/streamlitdatabase.csv", parse_dates=['timestamp'])
# st.table(data)
# Sidebar filters


left_column, right_column = st.columns((1,4))


# Add elements to the left column
with left_column:
    # Add more elements as needed

    x_axis = st.selectbox('Select X-axis', ['timestamp', 'bp_label', 'hour', 'weekday', 'patient_index'])
    y_axis = st.selectbox('Select Y-axis', ['timestamp', 'bp_label', 'hour', 'weekday', 'patient_index'])
    patient_filter = st.selectbox("Select a patient", list(range(1, 26)))

    y_low, y_up = st.slider('Select Y limit range',  0, 2400, (0, 12))
    x_low, x_up = st.slider('Select X limit range', 0, 2400, (0, 12))

# Add elements to the right column
with right_column:


# Filter the data based on the selected patient index
    filtered_data = data[data['patient_index'] == patient_filter]
# Create the scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(filtered_data[x_axis], filtered_data[y_axis], alpha=0.5,cmap="coolwarm")


    # the c parameter of the scatter method is set to df_filtered["bp_label"],
    #  which means the color of each data point will be determined by the bp_label column. 
    #  The cmap parameter is set to "coolwarm", which is a colormap 
    #  that ranges from cool colors (e.g. blue) for False values to warm colors (e.g. red) for True values.
    #   If you want to use a different color scheme, you can replace "coolwarm" with the name of another colormap.
    # To change the color of the scatter points themselves (as opposed to the color of the markers), 
    # you can add the edgecolors parameter to the scatter method with your desired color. 
    # For example, to make the points black, you can add edgecolors='black' to the scatter method.


    # Add tooltip with tweet text
    tooltip = plugins.PointHTMLTooltip(scatter, labels=list(filtered_data['tweet']))
    plugins.connect(fig, tooltip)

    # Format the plot
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Bipolar Label")



    plt.title(f'Scatter Plot of {x_axis} vs {y_axis} for Patient {patient_filter}')
    plt.ylim(y_low, y_up)
    plt.xlim(x_low, x_up)

    # Convert the plot to an interactive HTML
    html_graph = mpld3.fig_to_html(fig)

    # Display the interactive HTML graph
    components.html(html_graph, height=600)


l,r,m,p = st.columns((1,1,1,1))

with l:
    maxrow = st.slider('Maximum Rows to display',  0, 2400)

with m:
    timehrs = st.slider('Posted Hour of the day',  0, 24,(0,20))
    filtered_data = data[(data['hour'] >= timehrs[0]) & (data['hour'] <= timehrs[1])]
with r:
    weekda = st.slider('Posted day of the week',  0, 6,(0,6))
    filtered_data = data[(data['weekday'] >= weekda[0]) & (data['weekday'] <= weekda[1])]


with p:
    pl = st.radio('Bipolar label',['Tweets Labeled as Bipolar', 'Tweets that doesnt consist bipolar'] )

if pl == "Tweets Labeled as Bipolar":
    filtered_data = data[data['bp_label'] == True]
else:
    filtered_data = data[data['bp_label'] == False]

# filtered_data = data[data['patient_index'] == patient_filter]
st.table(filtered_data['tweet'].head(maxrow))