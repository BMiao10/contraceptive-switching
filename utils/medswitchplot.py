############# IMPORTS #############
### Common imports
# data 
import pandas as pd
import numpy as np

# system 
import os
import re
import datetime

# math and formatting
from scipy.stats import mannwhitneyu
#from umap import UMAP 

# plotting
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors

# NLP
from bertopic import BERTopic
import spacy

sns.set_style("white")
sns.set_context("talk")

"""
#import kaleido
#from bioinfokit import analys, visuz
#import pyodbc

# ML
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc, plot_roc_curve, classification_report
#import spacy
#import scispacy
#import spacy_transformers
"""

############# PREPROCESSING - QC #############

############# PREPROCESSING - PLOTTING #############

############# MEDICATION SWITCHING #############
def _formatSwitch(switch_df, index_dict, color_dict, switch=["1", "2"]):
    """
    Formats medication values in plotly Sankey happy format
    """
    switch_df = switch_df.groupby(switch).count().reset_index()

    switch_df["first_index"] = [index_dict[m] for m in switch_df[switch[0]]]
    switch_df["second_index"] = [index_dict[m] for m in switch_df[switch[1]]]
    switch_df["node_color"] = [color_dict[m] for m in switch_df[switch[0]]]
    switch_df["node_color"] = [c.replace("1)", "0.9)") for c in switch_df["node_color"]]
    switch_df["link_color"] = [c.replace("1)", "0.2)") for c in switch_df["node_color"]]
    
    return switch_df[["first_index", "second_index", "node_color", "link_color", "Count"]]

def _plotPatientTrajectorySankey(ra_meds_switch, labels, color_dict):
    """
    Actual plotting for Sankey
    """
    # sankey figure
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = list(color_dict.values()) #"grey"
        ),
        link = dict(
          source = ra_meds_switch["first_index"], # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = ra_meds_switch["second_index"],
          value = ra_meds_switch["Count"], 
            color = ra_meds_switch["node_color"]),  
        textfont = dict( family = "arial", size = 16))],
                    layout = go.Layout(autosize=False,width=850, height=650))

    # update names
    new_labels = [t.split("-")[1] if "-" in t else t for t in fig.data[0]["node"]["label"]]
    for trace in fig.data:
            trace.update(node={"label":new_labels}, visible=True)


    fig.update_layout(title_text="Medication switching", font_size=10)
    return fig


def plotPatientTrajectorySankey(ehr_df, time_col = 'startdatekeyvalue', patient_col= "patientdurablekey", 
                          values_col = "MedicationClass", switches=["1","2","3"], palette="pastel", save_fig=None):
    """
    Organize data for Plotly Sankey diagram plotting
    Plots number of patients in discrete values
    
    Params:
        time_col (str): column to sort values by
        patient_col (str): column containing patients to group by
        values_col (str): column containing values (eg. medication <str> or medication trajectory <list>) for Sankey labels
        switches (list<str>): Switches to plot
        palette (str, sns.palette): Seaborn palette
        
    Returns:
        Tuple<go.Figure, pd.DataFrame>

    """
    
    if type(ehr_df.iloc[0][values_col]) is str:
        # extract values
        ehr_df["groupedLabels"] = ehr_df[patient_col].map(extractInstance(ehr_df, patient_col=patient_col, n=None, 
                                                                      time_col=time_col, values_col=values_col))
    else:
        ehr_df["groupedLabels"] = ehr_df[values_col]

    # collapse values by patient
    ehr_df["nGroupedLabels"] = [m[:(int(switches[-1]))] if len(m)>(int(switches[-1])-1)  else m for m in ehr_df["groupedLabels"]]
    ehr_df = ehr_df.groupby(patient_col).first()
    
    # create labels and plot values
    labels = []
    for s in switches:
        ehr_df[s] = [(s)+"-"+m[int(s)-1] if len(m)>(int(s)-1) else s+"-No switch" for m in med_df["groupedLabels"]]
        labels.extend(list(ehr_df[s]))

    labels = list(set(labels))
    index_dict = dict(zip(labels, range(len(labels))))
    color_dict = dict(zip(index_dict, sns.color_palette(palette, len(index_dict)).as_hex()))
    
    # format switch for sankey
    ra_meds_switch = pd.DataFrame()
    ehr_df["Count"] = 1
    for i in range(len(switches)-1):
        curr_switch = _formatSwitch(ehr_df, index_dict, color_dict, switch=[switches[i], switches[i+1]])
        ra_meds_switch = pd.concat([ra_meds_switch, curr_switch])
        
    fig = _plotPatientTrajectorySankey(ra_meds_switch, labels, color_dict) 
    
    # add labels back
    index_dict_r = dict(zip(range(len(labels)), labels))
    ra_meds_switch["first_index_label"] = ra_meds_switch["first_index"].map(index_dict_r)
    ra_meds_switch["second_index_label"] = ra_meds_switch["second_index"].map(index_dict_r)
    return fig, ra_meds_switch

