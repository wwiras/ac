import json
import numpy as np


def load_graph_from_json(json_file_path):
    """
    Loads a graph from a JSON file and converts it into a weighted distance matrix.
    The JSON is expected to contain 'nodes' and 'edges' lists.
    The 'weight' from the edges is used as the distance.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing:
               - The number of nodes (int).
               - The distance matrix (np.array).
               - A dictionary mapping original node IDs to their index (int) in the matrix.
    """
    with open(json_file_path, 'r') as f:
        graph_data = json.load(f)

    # Extract node IDs and create a mapping to continuous indices
    node_ids = [node['id'] for node in graph_data['nodes']]
    num_nodes = len(node_ids)

    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    # Initialize the distance matrix with infinity
    # This represents unconnected nodes having an infinite distance
    distance_matrix = np.full((num_nodes, num_nodes), float('inf'))
    np.fill_diagonal(distance_matrix, 0)

    # Populate the matrix with edge weights
    for edge in graph_data['edges']:
        source_id = edge['source']
        target_id = edge['target']
        weight = edge['weight']

        source_index = id_to_index[source_id]
        target_index = id_to_index[target_id]

        # Since the graph is undirected, the matrix is symmetric
        distance_matrix[source_index, target_index] = weight
        distance_matrix[target_index, source_index] = weight

    return num_nodes, distance_matrix, id_to_index


def calculate_distance(cluster_a, cluster_b, distance_matrix):
    """
    Calculates the distance between two clusters using complete linkage (farthest neighbor).
    The distance between two clusters is defined as the maximum distance between
    any node in the first cluster and any node in the second cluster.
    """
    max_distance = 0.0
    for node_i in cluster_a:
        for node_j in cluster_b:
            distance = distance_matrix[node_i][node_j]
            if distance > max_distance:
                max_distance = distance

    return max_distance


def agglomerative_clustering(num_nodes, num_clusters, distance_matrix):
    """
    Performs agglomerative clustering based on the provided algorithm description.
    """
    # 9: Initialize C where each ci contains one node si.
    clusters = [[i] for i in range(num_nodes)]

    while len(clusters) > num_clusters:
        min_distance = float('inf')
        clusters_to_merge = None

        # 12: Find the pair of clusters with minimum D(ci, cj)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = calculate_distance(clusters[i], clusters[j], distance_matrix)

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = (i, j)

        if clusters_to_merge is not None:
            index_a, index_b = clusters_to_merge

            # Ensure the indices are sorted to avoid issues when removing
            if index_a > index_b:
                index_a, index_b = index_b, index_a

            # 13: Merge the two clusters
            new_cluster = clusters[index_a] + clusters[index_b]

            # 14: Update C: Remove old clusters and add the new one
            clusters.pop(index_b)
            clusters.pop(index_a)
            clusters.append(new_cluster)
        else:
            # Should not happen in a connected graph with more than 1 cluster
            break

    # 17: Return C
    return clusters


# Example Usage:
if __name__ == "__main__":
    # Specify the path to your JSON file
    json_file_path = 'nodes10_Jul262025010919_BA5.json'

    # Define the desired number of clusters (M).
    M = 3

    # Step 1: Load the graph and create the distance matrix
    num_nodes, distance_matrix, id_to_index = load_graph_from_json(json_file_path)

    # Step 2: Apply the agglomerative clustering algorithm
    clustered_result = agglomerative_clustering(num_nodes, M, distance_matrix)

    # Step 3: Map the results back to the original node IDs and output the clusters
    index_to_id = {v: k for k, v in id_to_index.items()}
    final_clusters_with_ids = [[index_to_id[node_index] for node_index in cluster] for cluster in clustered_result]

    print(f"Final Clusters (M = {M}):")
    for i, cluster in enumerate(final_clusters_with_ids):
        print(f"Cluster {i + 1}: {cluster}")