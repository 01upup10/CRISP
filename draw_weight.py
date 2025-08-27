import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio

def draw_histogram(tensor, save_dir="./output/pictures", save_name="hist", bins=15):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, save_name)
    tensor = np.array(tensor.cpu())
    plt.figure(figsize=(10, 6))
    sns.histplot(tensor, bins=bins, kde=True)
    plt.title(save_name.split('.')[0])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)

def draw_plot(tensor, save_name, save_dir="./output/pictures"):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, save_name)
    tensor = np.array(tensor.cpu())
    plt.figure(figsize=(10, 6))
    sns.lineplot(tensor)
    plt.title(save_name.split('.')[0])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(save_path)

def draw_weight(model_weights, save_dir, cur_epoch):
    
    x_len = model_weights.shape[0]
    if isinstance(model_weights, torch.Tensor):
        model_weights = model_weights.T.cpu().detach().numpy()
    else:
        model_weights = model_weights.T
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"weight_cur_{cur_epoch}.png")
    save_path_html = os.path.join(save_dir, f"weight_cur_{cur_epoch}.html")
    plt.figure(figsize=(20,10))
    plt.imshow(model_weights, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.title('Weights Matrix Heatmap')
    plt.xlabel('Classes')
    plt.ylabel('Features')
    plt.xticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    fig = go.Figure(data=go.Heatmap(
            z=model_weights[::-1],
            x=[f'{i}' for i in range(x_len)],
            y=[f'Feature {i}' for i in range(model_weights.shape[1])],
            colorscale='Viridis',
            colorbar=dict(title='Weight Value')
        ))

    fig.update_layout(
        title='Weights Matrix Heatmap',
        xaxis_title='Classes',
        yaxis_title='Features',
        xaxis=dict(tickangle=-90),
        yaxis=dict(showticklabels=False)
    )
    pio.write_html(fig, save_path_html)

def draw_cov(model_weights, save_dir, cur_epoch):

    x_len = model_weights.shape[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"cov_matrix_{cur_epoch}.png")
    save_path_html = os.path.join(save_dir, f"cov_matrix_{cur_epoch}.html")
    covariance_matrix = np.cov(model_weights.T.cpu().detach().numpy(), rowvar=False, bias=True)

    plt.figure(figsize=(20,10))
    plt.imshow(covariance_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Covariance Value')
    plt.title('Covariance Matrix Heatmap')
    plt.xlabel('Classes')
    plt.ylabel('Classes')
    plt.xticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
    plt.yticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    fig = go.Figure(data=go.Heatmap(
            z=covariance_matrix[::-1],
            x=[f'{i}' for i in range(x_len)],
            y=[f'Feature {i}' for i in range(covariance_matrix.shape[0])],
            colorscale='Viridis',
            colorbar=dict(title='Covariance Value')
        ))
    fig.update_layout(
        title='Covariance Heatmap',
        xaxis_title='Classes',
        yaxis_title='Features',
        xaxis=dict(tickangle=-90),
        yaxis=dict(showticklabels=False)
    )
    pio.write_html(fig, save_path_html)

def draw_correlation(model_weights, save_dir, cur_epoch):

    x_len = model_weights.shape[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"correlation_{cur_epoch}.png")
    save_path_html = os.path.join(save_dir, f"correlation_{cur_epoch}.html")
    p_matrix = np.corrcoef(model_weights.T.cpu().detach().numpy(), rowvar=False)
    plt.figure(figsize=(20,10))
    plt.imshow(p_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Correlation Value')
    plt.title('Pearson Correlation Matrix Heatmap')
    plt.xlabel('Classes')
    plt.ylabel('Classes')
    plt.xticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
    plt.yticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    fig = go.Figure(data=go.Heatmap(
            z=p_matrix[::-1],
            x=[f'{i}' for i in range(x_len)],
            y=[f'Feature {i}' for i in range(p_matrix.shape[1])],
            colorscale='Viridis',
            colorbar=dict(title='Correlation Value')
        ))
    fig.update_layout(
        title='Correlation Matrix Heatmap',
        xaxis_title='Classes',
        yaxis_title='Features',
        xaxis=dict(tickangle=-90),
        yaxis=dict(showticklabels=False)
    )

    pio.write_html(fig, save_path_html)

def draw_norm(model_weights, save_dir, cur_epoch):

    x_len = model_weights.shape[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"norm_{cur_epoch}.png")
    save_path_html = os.path.join(save_dir, f"norm_{cur_epoch}.html")

    weight_norm = np.linalg.norm(model_weights.T.cpu().detach().numpy(),axis=0)

    plt.figure(figsize=(20,10))
    plt.bar(range(weight_norm.shape[0]), weight_norm, color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('norm per Class')
    plt.xlabel('Classes')
    plt.ylabel('norm Value')
    plt.xticks(np.arange(x_len), [f'{i}' for i in range(x_len)])
    plt.tight_layout()
    plt.savefig(save_path)
    fig = go.Figure(data=go.Bar(x=list(range(0, len(weight_norm))), y=weight_norm))

    fig.update_layout(
        title='Norm Matrix Heatmap',
        xaxis_title='Classes',
        yaxis_title='Features',
        xaxis=dict(tickangle=-90),
        yaxis=dict(showticklabels=False)
    )

    pio.write_html(fig, save_path_html)

def draw_scores(scores, save_dir, cur_epoch):
    
    x_len = scores.shape[0]
    y_len = scores.shape[1]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"weight_cur_{cur_epoch}.png")
    save_path_html = os.path.join(save_dir, f"weight_cur_{cur_epoch}.html")
    plt.figure(figsize=(20,10))
    plt.imshow(scores.T.cpu().detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.title('Scores Matrix Heatmap')
    plt.xlabel('Queries')
    plt.ylabel('classes')
    plt.xticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
    plt.yticks(range(y_len))
    plt.tight_layout()
    plt.savefig(save_path)
    fig = go.Figure(data=go.Heatmap(
            z=scores.T.cpu().detach().numpy()[::-1],
            x=[f'{i}' for i in range(x_len)],
            y=[f'C{c}' for c in range(y_len)],
            colorscale='Viridis',
            colorbar=dict(title='Weight Value')
        ))

    fig.update_layout(
        title='Weights Matrix Heatmap',
        xaxis_title='Queries',
        yaxis_title='Classes',
        xaxis=dict(tickangle=-90),
        yaxis=dict(showticklabels=False)
    )

    pio.write_html(fig, save_path_html)