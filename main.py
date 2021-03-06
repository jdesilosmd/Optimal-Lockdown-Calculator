import numpy as np
import pandas as pd
from gekko import GEKKO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import base64


# Title
st.title('Optimal Lockdown Calculator')
st.markdown('###### **Beta Version:** 2.00')
st.markdown('##')
st.write('This web-based app aims to aid policymakers in identifying the optimal lockdown strength and duration needed\n'
         'to reduce transmission of the SARS-Cov-2 virus and at the same time, reducing the economic loss due to prolonged\n'
         'lockdowns. Social distancing measures can reduce the number of people infected, which in turn, can help hospitals\n'
         'operate within its effective capacity.')



# Sidebar
st.sidebar.markdown('## Data Input:')
location_in = st.sidebar.text_input(
    label='Type in the place you want to analyze:',
    value='NCR Plus'
)

st.sidebar.write('--------------')

st.sidebar.markdown('### Model Assumptions:')
N_in = st.sidebar.number_input(
    label='Type in N (total population):',
    min_value=1, value=28644207
)

Rt_in = st.sidebar.number_input(
    label='Type in the Effective Reproduction Number (R(t)):',
    value=1.24
)

t_incubation = st.sidebar.number_input(
    label='Type in the incubation period (in days):',
    value=5.1
)

t_infective = st.sidebar.number_input(
    label='Type in the infectious period (in days):',
    value=5
)

hcc_max = st.sidebar.number_input(
    label='Type in the actual or estimated maximum number of COVID-19 facilities in the target location (in percent of total population, N):',
    max_value=100.0,
    value=2.0
)

st.sidebar.write('--------------')
st.sidebar.markdown('### SEIR Model Figures')


i_in = st.sidebar.number_input(
    label='Type in the number of "INFECTED" people:',
    value=275842
)

e_in = st.sidebar.number_input(
    label='Type in the number of "EXPOSED" people:',
    value=i_in*5
)

r_in = st.sidebar.number_input(
    label='Type in the number of "RECOVERED" people:',
    value=376730
)

st.sidebar.write('Note: the SUSCEPTIBLE compartment will be computed automatically.')


def run_simulation(solver):
    # SEIR Model Computation using GEKKO:
    # fraction of infected and recovered individuals
    e_initial = e_in/N_in
    i_initial = i_in/N_in
    r_initial = r_in/N_in
    s_initial = 1 - e_initial - i_initial - r_initial

    alpha = 1/t_incubation
    gamma = 1/t_infective
    beta = Rt_in*gamma

    m = GEKKO()
    u = m.MV(0,lb=0.0,ub=0.9)

    s,e,i,r = m.Array(m.Var,4)
    s.value = s_initial
    e.value = e_initial
    i.value = i_initial
    r.value = r_initial
    m.Equations([s.dt()==-(1-u)*beta * s * i,\
                 e.dt()== (1-u)*beta * s * i - alpha * e,\
                 i.dt()==alpha * e - gamma * i,\
                 r.dt()==gamma*i])

    t = np.linspace(0, 30, 16)
#    t = np.insert(t, 1, [0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08, \
#                         0.2, 0.4, 0.8])
    m.time = t

    # initialize with simulation
    m.options.IMODE=7
    m.options.NODES=3
    m.options.MAX_ITER = 1000
    m.solve(disp=False)


    # plot the prediction
    predict_graph = make_subplots(rows=4, cols=1, subplot_titles=("<b>SEIR Model - No Lockdown:</b>"\
                                                                      " R(t) = {}, N = {}".format(Rt_in, N_in),
                                                                  "<b>Fraction of Susceptible and Recovered</b>",
                                                                  "<b>Fraction of Exposed and Infected</b>",
                                                                  "<b>Degree of Lockdown:</b> 0% = None, 100% = Full Lockdown"
                                                                  ))

    predict_graph.add_trace(go.Scatter(x=m.time, y=s.value, name='Susceptible',
                                       line=dict(color='teal', width=3)), row=1, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=r.value, name='Recovered',
                                       line=dict(color='firebrick', width=3)), row=1, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=i.value, name='Infectious',
                                       line=dict(color='orange', width=3)), row=1, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=e.value, name='Exposed',
                                       line=dict(color='purple', width=3)), row=1, col=1)
    predict_graph.update_xaxes(title_text="Time (Days)", row=1, col=1, zeroline=True, zerolinecolor='black')
    predict_graph.update_yaxes(title_text='Fraction', row=1, col=1, zeroline=True, zerolinecolor='black')



    predict_graph.add_trace(go.Scatter(x=m.time, y=s.value, showlegend=False,
                                       line=dict(color='teal', width=3)), row=2, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=r.value, showlegend=False,
                                       line=dict(color='firebrick', width=3)), row=2, col=1)


    predict_graph.add_trace(go.Scatter(x=m.time, y=i.value, showlegend=False,
                                       line=dict(color='orange', width=3)), row=3, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=e.value, showlegend=False,
                                       line=dict(color='purple', width=3)), row=3, col=1)

    # Optimize

    m.options.IMODE=6
    i.UPPER = hcc_max/100

    u.STATUS = 1
    m.options.SOLVER = solver
    m.options.TIME_SHIFT = 0
    s.value = s.value.value
    e.value = e.value.value
    i.value = i.value.value
    r.value = r.value.value
    m.Minimize(u)
    m.solve(disp=True)


    # Complete the plot
    predict_graph.add_trace(go.Scatter(x=m.time, y=s.value, name='Optimal Susceptible',
                                       line=dict(color='teal', dash='dot', width=2)), row=2, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=r.value, name='Optimal Recovered',
                                       line=dict(color='firebrick', dash='dot', width=2)), row=2, col=1)
    predict_graph.update_xaxes(title_text="Time (Days)", row=2, col=1, zeroline=True, zerolinecolor='black')
    predict_graph.update_yaxes(title_text='Fraction', row=2, col=1, zeroline=True, zerolinecolor='black')

    predict_graph.add_trace(go.Scatter(x=m.time, y=i.value, name='Infected (Below Hospital Capacity)',
                                       line=dict(color='orange', dash='dot', width=2)), row=3, col=1)
    predict_graph.add_trace(go.Scatter(x=m.time, y=e.value, name='Optimal Exposed',
                                       line=dict(color='purple', dash='dot', width=2)), row=3, col=1)
    predict_graph.update_xaxes(title_text="Time (Days)", row=3, col=1, zeroline=True, zerolinecolor='black')
    predict_graph.update_yaxes(title_text='Fraction', row=3, col=1, zeroline=True, zerolinecolor='black')


    d = {'Day': m.time, '% Lockdown Strength': u.value, '% Adjusted Lockdown Strength': 0.0}
    df_time = pd.DataFrame(data=d)
    df_time['% Lockdown Strength'] = df_time['% Lockdown Strength'].replace(0.0, 0.1)
    df_time['% Lockdown Strength'] = (df_time['% Lockdown Strength']*100)
    df_time['% Lockdown Strength'] = df_time['% Lockdown Strength'].round(2)
    df_time['% Adjusted Lockdown Strength'] = df_time['% Lockdown Strength']
    df_time['% Adjusted Lockdown Strength'].iloc[0:8] =df_time['% Lockdown Strength'].iloc[0:8].max()
    df_time['% Adjusted Lockdown Strength'].iloc[8:15] = df_time['% Lockdown Strength'].iloc[8:15].max()
    df_time['% Adjusted Lockdown Strength'].iloc[15:22] = df_time['% Lockdown Strength'].iloc[15:22].max()
    df_time['% Adjusted Lockdown Strength'].iloc[22:29] = df_time['% Lockdown Strength'].iloc[22:29].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[29:36] = df_time['% Lockdown Strength'].iloc[29:36].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[36:43] = df_time['% Lockdown Strength'].iloc[36:43].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[43:50] = df_time['% Lockdown Strength'].iloc[43:50].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[50:57] = df_time['% Lockdown Strength'].iloc[50:57].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[57:64] = df_time['% Lockdown Strength'].iloc[57:64].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[64:71] = df_time['% Lockdown Strength'].iloc[64:71].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[71:78] = df_time['% Lockdown Strength'].iloc[71:78].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[78:85] = df_time['% Lockdown Strength'].iloc[78:85].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[85:92] = df_time['% Lockdown Strength'].iloc[85:92].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[92:99] = df_time['% Lockdown Strength'].iloc[92:99].max()
#    df_time['% Adjusted Lockdown Strength'].iloc[99:101] = df_time['% Lockdown Strength'].iloc[99:101].max()
    df_time.Day = df_time.Day.astype(int)

    conditions = [
        (df_time['% Adjusted Lockdown Strength'] >= 75.0),
        (df_time['% Adjusted Lockdown Strength'] >= 50.0) & (df_time['% Lockdown Strength'] < 75.0),
        (df_time['% Adjusted Lockdown Strength'] >= 25.0) & (df_time['% Lockdown Strength'] < 50.0),
        (df_time['% Adjusted Lockdown Strength'] > 10.0) & (df_time['% Lockdown Strength'] < 25.0),
        (df_time['% Adjusted Lockdown Strength'] == 10.0)]

    choices = ['ECQ', 'MECQ', 'GCQ', 'MGCQ', 'N/A']
    df_time['Recommendation'] = np.select(conditions, choices, default='black')
#    df_time = df_time.iloc[8:]


    predict_graph.add_trace(go.Scatter(x=df_time['Day'], y=df_time['% Lockdown Strength'],
                                       name="Optimal Lockdown (0=None, 1=Full Lockdown)",
                                       mode='lines', line=dict(color='red', width=2, shape='vh')), row=4, col=1)
    predict_graph.add_trace(go.Scatter(x=df_time['Day'], y=df_time['% Adjusted Lockdown Strength'],
                                       name="14-day Adjusted Optimal Lockdown",
                                       mode='lines', line=dict(color='black', dash='dot',
                                                               width=2, shape='vh')), row=4, col=1)
    predict_graph.update_xaxes(title_text="Time (Days)", row=4, col=1, zeroline=True,
                               range=[0, 30], zerolinecolor='black')
    predict_graph.update_yaxes(title_text='% Lockdown Strength', row=4, col=1, zeroline=True, zerolinecolor='black')


    predict_graph.add_hrect(y0=75, y1=100, fillcolor="orange", opacity=0.25,
                            layer="below", line_width=0, annotation_text="<b>ECQ</b>", annotation_position="top right",
                            row=4, col=1)

    predict_graph.add_hrect(y0=50, y1=75, fillcolor="yellow", opacity=0.25,
                            layer="below", line_width=0, annotation_text="<b>MECQ</b>", annotation_position="top right",
                            row=4, col=1)

    predict_graph.add_hrect(y0=25, y1=50, fillcolor="palegreen", opacity=0.25,
                            layer="below", line_width=0, annotation_text="<b>GCQ</b>", annotation_position="top right",
                            row=4, col=1)

    predict_graph.add_hrect(y0=0, y1=25, fillcolor="lime", opacity=0.25,
                            layer="below", line_width=0, annotation_text="<b>MGCQ</b>", annotation_position="top right",
                            row=4, col=1)


    predict_graph.update_layout(template='seaborn', autosize=False, width=1000, height=1500,
                                title_text='<b>Optimum Lockdown Model for {} (at Health Care Capacity = {}% of N)</b>'.format(location_in, hcc_max),
                                legend=dict(orientation="h"))

    st.plotly_chart(predict_graph, use_container_width=True)


    # Generate a table of results

    st.markdown('### Detailed table of recommended lockdown strength adjustment every 2 days and 14 days (for 30 days):')
    df_time_table = go.Figure(data=[go.Table(
        header=dict(values=list(df_time.columns),
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[df_time['Day'], df_time['% Lockdown Strength'],
                           df_time['% Adjusted Lockdown Strength'], df_time['Recommendation']],
                   fill_color='azure',
                   align='left'))
    ])

    st.plotly_chart(df_time_table, use_container_width=True)

    def get_table_download_link_csv(df):
        csv_export = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv_export).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="Optimum_Lockdown.csv" target="_blank">Download csv file</a>'
        return href

    st.markdown(get_table_download_link_csv(df_time), unsafe_allow_html=True)


#    raise Exception ('Error: Solution not found. Try increasing the maximum number of health facilities in the sidebar.')

st.markdown('##')
st.markdown('##')

solver_radio = st.radio(
    'Select an equation solver',
    ('Solver 1: Advanced Process Optimizer (APOPT)',
     'Solver 2: Interior Point Optimizer (IPOPT)'))

if solver_radio == 'Solver 1: Advanced Process Optimizer (APOPT)':
    if st.button(label='Run Simulation'):
        with st.spinner('Simulating...'):
            try:
                run_simulation(1)

            except:
                st.error('Error: Solution not found. Try increasing the maximum number of health facilities in the sidebar.')
    else:
        st.write("Click button to run the simulation.")

else:
    if st.button(label='Run Simulation'):
        with st.spinner('Simulating...'):
            try:
                run_simulation(3)

            except:
                st.error(
                    'Error: Solution not found. Try increasing the maximum number of health facilities in the sidebar.')
    else:
        st.write("Click button to run the simulation.")
