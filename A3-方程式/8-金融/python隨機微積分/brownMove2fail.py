import plotly.graph_objs as go

# 使用Plotly可視化布朗運動
def plot_brownian_motion(brownian_motion):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.linspace(0, T, N),
        y=brownian_motion,
        mode='lines',
        name='Brownian Motion'
    ))
    fig.update_layout(
        title='Interactive Visualization of Brownian Motion',
        xaxis_title='Time',
        yaxis_title='B(t)',
        showlegend=True
    )
    fig.show()

# 可視化布朗運動
plot_brownian_motion(brownian_motion)
