import plotly.graph_objects as go

node_labels = ["Steady recovery", "Steady decline", "Early recovery with chronic decline", 
               "Late recovery with acute decline", "INFARCT", "BLEEDING", "OTHER"]

sankey_trace = go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
    ),
    link=dict(
        source=[0, 1, 2, 3],
        target=[4, 5, 6, 4],
        value=[10, 15, 5, 20]
    )
)

fig = go.Figure(data=[sankey_trace])
fig.update_layout(title_text="Test Sankey Diagram", width=800, height=600)
fig.write_image("test_sankey.svg", engine="kaleido")
