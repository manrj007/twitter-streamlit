from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
import streamlit as st



class_names = ['positive', 'negative', 'neutral']
cnt= [28.0,56.0,16.0]

chart_data = pd.DataFrame(columns=["label", "count"])
chart_data['label'] = class_names
chart_data['count'] = cnt
st.bar_chart(chart_data)
