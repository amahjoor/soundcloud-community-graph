import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from scraper import SoundCloudScraper
import json

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SoundCloud Ego Network Visualizer')
    
    parser.add_argument('account', type=str, help='SoundCloud account username to visualize')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to store/load data')
    parser.add_argument('--browser', type=str, choices=['safari', 'chrome'], default='safari',
                       help='Browser to use for scraping (default: safari)')
    parser.add_argument('--max_items', type=int, default=500,
                       help='Maximum number of items to collect per interaction type')
    parser.add_argument('--min_weight', type=int, default=1,
                       help='Minimum interaction weight to include in visualization')
    parser.add_argument('--skip_scrape', action='store_true',
                       help='Skip scraping and use existing data')
    
    return parser.parse_args()

def scrape_account(account, data_dir, browser, max_items):
    """Scrape a single account's interactions."""
    print(f"\n====== Scraping Account: {account} ======\n")
    
    # Initialize the scraper
    scraper = SoundCloudScraper(
        data_dir=data_dir,
        max_items=max_items,
        browser=browser
    )
    
    # Create a directory for the ego network if it doesn't exist
    ego_dir = os.path.join(data_dir, "ego_networks", account)
    os.makedirs(ego_dir, exist_ok=True)
    
    # Collect interactions for the account
    success, account_data, interaction_data = scraper.collect_interactions(account)
    
    if not success:
        print(f"Failed to scrape account: {account}")
        return None, None
    
    # Process the interaction data
    all_interactions = []
    
    # Process each interaction type with appropriate weights
    for interaction_type, data in interaction_data.items():
        weight = 1 if interaction_type == 'likes' else 3
        
        for i in range(data.shape[0]):
            target_account = data[i, 1]
            all_interactions.append({
                'source': account,
                'target': target_account,
                'weight': weight,
                'type': interaction_type
            })
    
    # Create a DataFrame with all interactions
    interactions_df = pd.DataFrame(all_interactions)
    
    # Save the interactions to a CSV file
    interactions_file = os.path.join(ego_dir, "interactions.csv")
    interactions_df.to_csv(interactions_file, index=False)
    
    # Save account data
    account_file = os.path.join(ego_dir, "account_info.csv")
    pd.DataFrame([account_data]).to_csv(account_file, index=False)
    
    print(f"Scraped {len(interactions_df)} interactions for {account}")
    
    return account_data, interactions_df

def load_existing_data(account, data_dir):
    """Load existing data for an account."""
    ego_dir = os.path.join(data_dir, "ego_networks", account)
    
    # Check if the data exists
    interactions_file = os.path.join(ego_dir, "interactions.csv")
    account_file = os.path.join(ego_dir, "account_info.csv")
    
    if not os.path.exists(interactions_file) or not os.path.exists(account_file):
        print(f"No existing data found for account: {account}")
        return None, None
    
    # Load the data
    interactions_df = pd.read_csv(interactions_file)
    account_data = pd.read_csv(account_file).iloc[0].to_dict()
    
    print(f"Loaded {len(interactions_df)} interactions for {account}")
    
    return account_data, interactions_df

def visualize_ego_network(account, account_data, interactions_df, data_dir, min_weight=1):
    """Create visualizations of the ego network."""
    print(f"\n====== Visualizing Ego Network for: {account} ======\n")
    
    if interactions_df is None or account_data is None:
        print("No data available for visualization")
        return
    
    # Filter interactions based on minimum weight
    if min_weight > 0:
        interactions_df = interactions_df[interactions_df['weight'] >= min_weight]
    
    # Create output directory
    ego_dir = os.path.join(data_dir, "ego_networks", account)
    os.makedirs(ego_dir, exist_ok=True)
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add the ego node (central account)
    G.add_node(account, type='ego', followers=account_data['followers'])
    
    # Add edges and nodes from interactions
    for _, row in interactions_df.iterrows():
        if row['target'] not in G:
            G.add_node(row['target'], type='alter')
        
        # If the edge already exists, increase the weight
        if G.has_edge(row['source'], row['target']):
            G[row['source']][row['target']]['weight'] += row['weight']
        else:
            G.add_edge(row['source'], row['target'], weight=row['weight'], type=row['type'])
    
    # Get position layout using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Create a static visualization
    plt.figure(figsize=(16, 12))
    
    # Draw the ego node with a distinctive color and size
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[account],
        node_color='red',
        node_size=1000,
        alpha=0.8
    )
    
    # Draw alter nodes with color based on interaction type
    interaction_types = interactions_df['type'].unique()
    
    for i, interaction_type in enumerate(interaction_types):
        # Get nodes that have this interaction type
        nodes = interactions_df[interactions_df['type'] == interaction_type]['target'].unique()
        
        # Draw these nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=f'C{i+1}',
            node_size=300,
            alpha=0.7
        )
    
    # Draw edges with width based on weight
    edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.4
    )
    
    # Draw labels for the ego node and some top alter nodes
    labels = {account: account}
    top_alters = interactions_df.groupby('target')['weight'].sum().nlargest(15).index
    for alter in top_alters:
        labels[alter] = alter
        
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_family='sans-serif'
    )
    
    # Add a legend for interaction types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=15, label=f'{account} (Ego)')]
    
    for i, interaction_type in enumerate(interaction_types):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=f'C{i+1}', 
                       markersize=10, label=f'{interaction_type.capitalize()}')
        )
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'SoundCloud Ego Network: {account}')
    plt.axis('off')
    
    # Save the static visualization
    static_file = os.path.join(ego_dir, f"{account}_network.png")
    plt.savefig(static_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Static visualization saved to {static_file}")
    
    # Create an interactive visualization with Plotly
    edge_traces = []
    
    # Create a trace for each interaction type
    for interaction_type in interaction_types:
        edges = [(u, v) for u, v in G.edges() if G[u][v].get('type') == interaction_type]
        
        if not edges:
            continue
        
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[u][v]['weight'])
        
        color = {
            'likes': 'rgba(31, 119, 180, 0.6)',
            'comments': 'rgba(255, 127, 14, 0.6)',
            'following': 'rgba(44, 160, 44, 0.6)',
            'reposts': 'rgba(214, 39, 40, 0.6)'
        }.get(interaction_type, 'rgba(128, 128, 128, 0.6)')
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color=color),
            hoverinfo='none',
            mode='lines',
            name=interaction_type.capitalize()
        )
        
        edge_traces.append(edge_trace)
    
    # Create a trace for nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Text for hover info
        if node == account:
            node_text.append(f"{node} (Ego)<br>Followers: {account_data['followers']}")
            node_size.append(20)
            node_color.append('red')
        else:
            interactions = interactions_df[interactions_df['target'] == node]
            total_weight = interactions['weight'].sum()
            types = interactions['type'].unique()
            
            node_text.append(f"{node}<br>Total Interaction Weight: {total_weight}<br>Types: {', '.join(types)}")
            node_size.append(10)
            node_color.append('lightblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='black')
        ),
        name='Accounts'
    )
    
    # Create the figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f'SoundCloud Ego Network: {account}',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=900,
            height=700
        )
    )
    
    # Save the interactive visualization with advanced search and filtering
    interactive_file = os.path.join(ego_dir, f"{account}_network_interactive.html")
    
    # Create node and edge data for JavaScript
    node_data = []
    for i, node in enumerate(G.nodes()):
        node_info = {
            'id': node,
            'x': pos[node][0],
            'y': pos[node][1],
            'size': node_size[i],
            'color': node_color[i],
            'text': node_text[i],
            'isEgo': node == account
        }
        node_data.append(node_info)
    
    edge_data = []
    for i, interaction_type in enumerate(interaction_types):
        edges = [(u, v) for u, v in G.edges() if G[u][v].get('type') == interaction_type]
        for u, v in edges:
            edge_info = {
                'source': u,
                'target': v,
                'type': interaction_type,
                'weight': G[u][v]['weight']
            }
            edge_data.append(edge_info)
    
    # Export node and edge data as JSON for JavaScript
    node_data_json = json.dumps(node_data)
    edge_data_json = json.dumps(edge_data)
    
    # Create HTML with embedded JavaScript for search and filtering
    with open(interactive_file, 'w') as f:
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>SoundCloud Ego Network: {account}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
        .container {{ display: flex; height: 100vh; }}
        .sidebar {{ 
            width: 250px; background-color: #f8f8f8; padding: 20px; 
            border-right: 1px solid #ddd; overflow-y: auto;
            flex-shrink: 0;
        }}
        .main-content {{ 
            flex-grow: 1; 
            position: relative; 
            width: calc(100% - 290px);
            height: 100%;
        }}
        #graph {{ 
            width: 100%; 
            height: 100%; 
        }}
        h2 {{ margin-top: 0; font-size: 18px; }}
        .filter-section {{ margin-bottom: 20px; }}
        input[type="text"] {{ 
            width: 100%; padding: 8px; margin-bottom: 10px;
            border: 1px solid #ddd; border-radius: 4px;
        }}
        .checkbox-item {{ margin-bottom: 5px; }}
        button {{ 
            padding: 8px 12px; background-color: #4CAF50; color: white;
            border: none; border-radius: 4px; cursor: pointer;
            margin-right: 5px; margin-bottom: 5px;
        }}
        button:hover {{ background-color: #45a049; }}
        button.reset {{ background-color: #f44336; }}
        button.reset:hover {{ background-color: #d32f2f; }}
        .stats {{ margin-top: 20px; }}
        .stats p {{ margin: 5px 0; font-size: 14px; }}
        #nodeCount, #edgeCount {{ font-weight: bold; }}
        #loading {{ display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>SoundCloud Ego Network</h2>
            <div class="filter-section">
                <h3>Search</h3>
                <input type="text" id="searchInput" placeholder="Search accounts...">
                <div id="searchResults"></div>
                <button id="clearSearch">Clear Search</button>
            </div>
            
            <div class="filter-section">
                <h3>Filter by Interaction Type</h3>
                <div class="checkbox-item">
                    <input type="checkbox" id="likesFilter" checked>
                    <label for="likesFilter">Likes</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="commentsFilter" checked>
                    <label for="commentsFilter">Comments</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="followingFilter" checked>
                    <label for="followingFilter">Following</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="repostsFilter" checked>
                    <label for="repostsFilter">Reposts</label>
                </div>
            </div>
            
            <div class="filter-section">
                <h3>Actions</h3>
                <button id="applyFilters">Apply Filters</button>
                <button id="resetGraph" class="reset">Reset Graph</button>
            </div>
            
            <div class="stats">
                <h3>Network Statistics</h3>
                <p>Ego: <strong>{account}</strong></p>
                <p>Followers: <strong>{account_data['followers']}</strong></p>
                <p>Nodes: <span id="nodeCount">{G.number_of_nodes()}</span></p>
                <p>Edges: <span id="edgeCount">{G.number_of_edges()}</span></p>
                <div id="loading">Loading...</div>
            </div>
        </div>
        
        <div class="main-content">
            <div id="graph"></div>
        </div>
    </div>

    <script>
        // Node and edge data
        const allNodes = {node_data_json};
        const allEdges = {edge_data_json};
        
        // Current state
        let currentNodes = [...allNodes];
        let currentEdges = [...allEdges];
        let graph = null;
        let searchTerm = '';
        
        // Colors for different interaction types
        const colors = {{
            'likes': 'rgba(31, 119, 180, 0.6)',
            'comments': 'rgba(255, 127, 14, 0.6)',
            'following': 'rgba(44, 160, 44, 0.6)',
            'reposts': 'rgba(214, 39, 40, 0.6)'
        }};
        
        // Create the initial graph
        function createGraph() {{
            const edgeTraces = createEdgeTraces(currentEdges);
            const nodeTrace = createNodeTrace(currentNodes);
            
            const layout = {{
                title: 'SoundCloud Ego Network: {account}',
                showlegend: true,
                hovermode: 'closest',
                margin: {{b: 20, l: 5, r: 5, t: 40}},
                xaxis: {{showgrid: false, zeroline: false, showticklabels: false}},
                yaxis: {{showgrid: false, zeroline: false, showticklabels: false}},
                width: document.querySelector('.main-content').offsetWidth,
                height: document.querySelector('.main-content').offsetHeight - 20,
                autosize: true
            }};
            
            graph = Plotly.newPlot('graph', [...edgeTraces, nodeTrace], layout);
            
            // Update stats
            document.getElementById('nodeCount').textContent = currentNodes.length;
            document.getElementById('edgeCount').textContent = currentEdges.length;
            
            // Handle window resize
            window.addEventListener('resize', function() {{
                Plotly.relayout('graph', {{
                    width: document.querySelector('.main-content').offsetWidth,
                    height: document.querySelector('.main-content').offsetHeight - 20
                }});
            }});
        }}
        
        // Create edge traces by interaction type
        function createEdgeTraces(edges) {{
            const edgesByType = {{}};
            
            // Group edges by type
            edges.forEach(edge => {{
                if (!edgesByType[edge.type]) {{
                    edgesByType[edge.type] = [];
                }}
                edgesByType[edge.type].push(edge);
            }});
            
            const traces = [];
            
            // Create a trace for each type
            for (const type in edgesByType) {{
                const edgeList = edgesByType[type];
                const x = [];
                const y = [];
                
                edgeList.forEach(edge => {{
                    const source = allNodes.find(n => n.id === edge.source);
                    const target = allNodes.find(n => n.id === edge.target);
                    
                    if (source && target) {{
                        x.push(source.x, target.x, null);
                        y.push(source.y, target.y, null);
                    }}
                }});
                
                const trace = {{
                    x: x,
                    y: y,
                    mode: 'lines',
                    line: {{
                        width: 1,
                        color: colors[type] || 'rgba(128, 128, 128, 0.6)'
                    }},
                    hoverinfo: 'none',
                    name: type.charAt(0).toUpperCase() + type.slice(1)
                }};
                
                traces.push(trace);
            }}
            
            return traces;
        }}
        
        // Create node trace
        function createNodeTrace(nodes) {{
            const x = [];
            const y = [];
            const text = [];
            const sizes = [];
            const colors = [];
            
            nodes.forEach(node => {{
                x.push(node.x);
                y.push(node.y);
                text.push(node.text);
                sizes.push(node.size);
                colors.push(node.color);
            }});
            
            return {{
                x: x,
                y: y,
                mode: 'markers',
                hoverinfo: 'text',
                text: text,
                marker: {{
                    size: sizes,
                    color: colors,
                    line: {{
                        width: 1,
                        color: 'black'
                    }}
                }},
                name: 'Accounts'
            }};
        }}
        
        // Apply search filter
        function applySearch() {{
            searchTerm = document.getElementById('searchInput').value.toLowerCase().trim();
            
            if (!searchTerm) {{
                return false;
            }}
            
            // Find matching nodes
            const matchingNodes = allNodes.filter(node => 
                node.text.toLowerCase().includes(searchTerm) || node.id.toLowerCase().includes(searchTerm)
            );
            
            // If no matches, alert user
            if (matchingNodes.length === 0) {{
                alert('No accounts found matching "' + searchTerm + '"');
                return false;
            }}
            
            // Show only matching nodes and the ego node
            currentNodes = allNodes.filter(node => 
                matchingNodes.some(match => match.id === node.id) || node.isEgo
            );
            
            // Show only edges connected to matching nodes
            currentEdges = allEdges.filter(edge => 
                currentNodes.some(node => node.id === edge.source) && 
                currentNodes.some(node => node.id === edge.target)
            );
            
            // Also apply type filters
            applyTypeFilters();
            
            return true;
        }}
        
        // Apply interaction type filters
        function applyTypeFilters() {{
            const showLikes = document.getElementById('likesFilter').checked;
            const showComments = document.getElementById('commentsFilter').checked;
            const showFollowing = document.getElementById('followingFilter').checked;
            const showReposts = document.getElementById('repostsFilter').checked;
            
            let filteredEdges;
            
            // If we have a search term active, filter from current edges
            if (searchTerm) {{
                filteredEdges = currentEdges.filter(edge => {{
                    if (edge.type === 'likes' && !showLikes) return false;
                    if (edge.type === 'comments' && !showComments) return false;
                    if (edge.type === 'following' && !showFollowing) return false;
                    if (edge.type === 'reposts' && !showReposts) return false;
                    return true;
                }});
            }} else {{
                // Otherwise, start fresh from all edges
                filteredEdges = allEdges.filter(edge => {{
                    if (edge.type === 'likes' && !showLikes) return false;
                    if (edge.type === 'comments' && !showComments) return false;
                    if (edge.type === 'following' && !showFollowing) return false;
                    if (edge.type === 'reposts' && !showReposts) return false;
                    return true;
                }});
            }}
            
            currentEdges = filteredEdges;
            
            // Update nodes to show only those connected by edges plus ego
            const connectedNodeIds = new Set();
            connectedNodeIds.add('{account}'); // Always include ego node
            
            currentEdges.forEach(edge => {{
                connectedNodeIds.add(edge.source);
                connectedNodeIds.add(edge.target);
            }});
            
            currentNodes = allNodes.filter(node => connectedNodeIds.has(node.id));
            
            return true;
        }}
        
        // Reset graph to initial state
        function resetGraph() {{
            document.getElementById('searchInput').value = '';
            document.getElementById('likesFilter').checked = true;
            document.getElementById('commentsFilter').checked = true;
            document.getElementById('followingFilter').checked = true;
            document.getElementById('repostsFilter').checked = true;
            
            searchTerm = '';
            currentNodes = [...allNodes];
            currentEdges = [...allEdges];
            
            updateGraph();
        }}
        
        // Update the graph with current filters
        function updateGraph() {{
            document.getElementById('loading').style.display = 'block';
            
            setTimeout(() => {{
                const edgeTraces = createEdgeTraces(currentEdges);
                const nodeTrace = createNodeTrace(currentNodes);
                
                Plotly.react('graph', [...edgeTraces, nodeTrace], {{
                    title: 'SoundCloud Ego Network: {account}',
                    showlegend: true,
                    hovermode: 'closest',
                    margin: {{b: 20, l: 5, r: 5, t: 40}},
                    xaxis: {{showgrid: false, zeroline: false, showticklabels: false}},
                    yaxis: {{showgrid: false, zeroline: false, showticklabels: false}},
                    width: document.querySelector('.main-content').offsetWidth,
                    height: document.querySelector('.main-content').offsetHeight - 20
                }});
                
                // Update stats
                document.getElementById('nodeCount').textContent = currentNodes.length;
                document.getElementById('edgeCount').textContent = currentEdges.length;
                
                document.getElementById('loading').style.display = 'none';
            }}, 50);
        }}
        
        // Event listeners
        document.getElementById('applyFilters').addEventListener('click', () => {{
            applyTypeFilters();
            updateGraph();
        }});
        
        document.getElementById('resetGraph').addEventListener('click', resetGraph);
        
        document.getElementById('clearSearch').addEventListener('click', () => {{
            document.getElementById('searchInput').value = '';
            searchTerm = '';
            
            // Reset to all nodes then apply type filters
            currentNodes = [...allNodes];
            applyTypeFilters();
            updateGraph();
        }});
        
        document.getElementById('searchInput').addEventListener('keyup', (e) => {{
            if (e.key === 'Enter') {{
                if (applySearch()) {{
                    updateGraph();
                }}
            }}
        }});
        
        // Initialize graph
        document.addEventListener('DOMContentLoaded', createGraph);
    </script>
</body>
</html>
        '''
        
        f.write(html_content)
    
    print(f"Interactive visualization with search and filtering saved to {interactive_file}")
    
    # Print some basic network statistics
    print("\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Print top connected accounts
    top_connections = interactions_df.groupby('target')['weight'].sum().nlargest(10)
    print("\nTop 10 connections:")
    for target, weight in top_connections.items():
        print(f"  {target}: {weight}")
    
    return G

def main():
    """Main function to run the ego network visualization."""
    args = parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Create ego networks directory
    ego_networks_dir = os.path.join(args.data_dir, "ego_networks")
    os.makedirs(ego_networks_dir, exist_ok=True)
    
    if args.skip_scrape:
        # Load existing data
        account_data, interactions_df = load_existing_data(args.account, args.data_dir)
    else:
        # Scrape the account
        account_data, interactions_df = scrape_account(
            args.account,
            args.data_dir,
            args.browser,
            args.max_items
        )
    
    # Visualize the ego network
    if account_data is not None and interactions_df is not None:
        visualize_ego_network(
            args.account,
            account_data,
            interactions_df,
            args.data_dir,
            args.min_weight
        )
    else:
        print(f"Could not visualize ego network for {args.account} due to missing data")

if __name__ == "__main__":
    main() 