import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_site(sitename, y_test, y_pred, r2):

    fig = make_subplots(rows=2, cols=1, subplot_titles=['GPP site: ' + sitename + "     (R2: "+ str(np.around(r2, decimals=2)) + ")", "Residuals site: "  + sitename])
    xs = np.arange(y_test.size)

    fig.add_trace(go.Scatter(x=xs, y=y_test,
                        mode='markers',
                        name='Ground Truth', 
                        marker=dict(size=4,
                                    color = "light blue"
                                    )),
                        row=1,
                        col=1
    )

    fig.add_trace(go.Scatter(x=xs, y=y_pred,
                        mode='markers',
                        name='Prediction', marker=dict(size=4,
                                                    color="darksalmon")), 
                row=1,
                col=1
                
    )

    fig.add_trace(go.Scatter(x=xs, y=(y_pred-y_test)**2,
                        mode='markers',
                        name='Residual', marker=dict(size=4,
                                                    color="cadetblue")), 
                row=2,
                col=1)
        

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="GPP", row=1, col=1)

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_layout(title_font_size=30)
    fig.write_html( "plots/" + sitename + ".html")