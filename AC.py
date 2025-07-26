import json
import numpy as np
import argparse
import os


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

    Returns float('inf') if no valid (finite, non-self-loop) connections exist
    between the two clusters.
    """
    # Initialize current_max_distance to negative infinity to correctly find the maximum
    # among positive distances, or ensure it remains -inf if no valid connections.
    current_max_distance = float('-inf')
    found_valid_connection = False

    for node_i in cluster_a:
        for node_j in cluster_b:
            # Skip if comparing a node to itself (distance 0)
            if node_i == node_j:
                continue

            distance = distance_matrix[node_i][node_j]

            # Skip if there's no connection (distance inf)
            if distance == float('inf'):
                continue

            # If we reach here, 'distance' is a valid, finite, non-self-loop distance
            found_valid_connection = True
            if distance > current_max_distance:
                current_max_distance = distance

    # If no valid connections were found between the two clusters,
    # it means they are effectively infinitely far apart for complete linkage.
    if not found_valid_connection:
        return float('inf')
    else:
        return current_max_distance


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
            for j in range(i + 1, len(clusters)):  # Iterate through unique pairs
                distance = calculate_distance(clusters[i], clusters[j], distance_matrix)

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = (i, j)

        if clusters_to_merge is not None:
            index_a, index_b = clusters_to_merge

            # Ensure the indices are sorted to avoid issues when removing
            # Remove the higher index first to avoid shifting issues
            if index_a > index_b:
                index_a, index_b = index_b, index_a

            # 13: Merge the two clusters
            new_cluster = clusters[index_a] + clusters[index_b]

            # 14: Update C: Remove old clusters and add the new one
            clusters.pop(index_b)
            clusters.pop(index_a)
            clusters.append(new_cluster)
        else:
            # This case should only be reached if there are no more possible merges
            # (e.g., all remaining clusters are infinitely far apart, which shouldn't
            # happen if the graph is connected or if M is not 1)
            print("Warning: No more clusters could be merged. Remaining clusters might be disconnected.")
            break

    # 17: Return C
    return clusters


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Perform Agglomerative Clustering on a network graph from a JSON file.")

    # Change arguments to use flags
    parser.add_argument("--json_file_path", type=str, help="Path to the JSON file containing the graph topology.",
                        required=True)
    parser.add_argument("--num_clusters", type=int, help="Desired number of clusters (M).", required=True)

    # Parse command-line arguments
    args = parser.parse_args()

    # json_file_path = args.json_file_path
    json_file_path = os.path.join("topology",args.json_file_path)
    print(f"Loading network from : {json_file_path}")
    M = args.num_clusters
    print(f"Total cluster(s), M : {M}")

    # Check for the existence of the JSON file before proceeding
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'")
        exit(1)  # Exit the script with an error code

    # Validate M
    if M <= 0:
        print("Error: The desired number of clusters (M) must be a positive integer.")
        exit(1)

    try:
        # Step 1: Load the graph and create the distance matrix
        num_nodes, distance_matrix, id_to_index = load_graph_from_json(json_file_path)

        # Additional check: M should not be greater than the number of nodes
        if M > num_nodes:
            print(
                f"Warning: Desired number of clusters (M={M}) is greater than the total number of nodes ({num_nodes}).")
            print(f"Clustering will result in {num_nodes} clusters, where each node is its own cluster.")
            M = num_nodes  # Adjust M to prevent an infinite loop or unexpected behavior

        print("--- Loaded Distance Matrix ---")
        # Print the matrix with a clear format
        # Use a custom formatter for better readability of large matrices
        np.set_printoptions(linewidth=np.inf, precision=2, suppress=True,
                            formatter={'float_kind': lambda x: "inf" if np.isinf(x) else f"{x:.0f}"})
        print(distance_matrix)
        print("\n")

        # Step 2: Apply the agglomerative clustering algorithm
        clustered_result = agglomerative_clustering(num_nodes, M, distance_matrix)

        # Step 3: Map the results back to the original node IDs and output the clusters
        index_to_id = {v: k for k, v in id_to_index.items()}
        final_clusters_with_ids = [[index_to_id[node_index] for node_index in cluster] for cluster in clustered_result]

        print(f"--- Final Clusters (M = {M}) ---")
        for i, cluster in enumerate(final_clusters_with_ids):
            print(f"Cluster {i + 1}: {cluster}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Please ensure it's a valid JSON file.")
        exit(1)
    except KeyError as e:
        print(f"Error: Missing expected key in JSON file: {e}. Please check the JSON structure.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)