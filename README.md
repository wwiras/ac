# Agglomerative Clustering
Agglomerative Clustering with BNSF

### Step 1 - Create a network topology (based on ER or BA)
Below is the example on how to create a network overlay of BA/ER network.

```shell
# Create overlay networkDeployments. Change the values accordingly
$ python network_constructor.py --model BA --nodes 1000 --others 4
Initial status from the input .....
Number of nodes in the network: 1000
Average neighbor (degree): 4
Creating BARABASI ALBERT (BA) network model .....
Average degree: 7.968
Target degree:4
nx.is_connected(network): True
Graph Before: Graph with 1000 nodes and 3984 edges
BA network model is SUCCESSFUL ! ....
Graph After: Graph with 1000 nodes and 3984 edges
Do you want to save the graph? (y/n): y
Topology saved to nodes1000_May172025154908_BA4.json
```

### Step 2 - Apply Agglomerative Clustering [based on BA topology (json) and M total clusters ] 

```shell
python AC.py --json_file_path nodes10_Jul262025125801_BA5.json --num_clusters 3 
Loading network from : topology/nodes10_Jul262025125801_BA5.json
Total cluster(s), M : 3
--- Loaded Distance Matrix ---
[[0 96 45 92 14 95 73 10 inf 36]
 [96 0 inf inf inf inf 19 38 35 inf]
 [45 inf 0 inf inf inf 71 inf inf 48]
 [92 inf inf 0 inf inf 84 90 59 80]
 [14 inf inf inf 0 inf inf inf 60 inf]
 [95 inf inf inf inf 0 60 23 inf 66]
 [73 19 71 84 inf 60 0 34 59 35]
 [10 38 inf 90 inf 23 34 0 52 inf]
 [inf 35 inf 59 60 inf 59 52 0 inf]
 [36 inf 48 80 inf 66 35 inf inf 0]]


--- Final Clusters (M = 3) ---
Cluster 1: ['gossip-2', 'gossip-4', 'gossip-0', 'gossip-7']
Cluster 2: ['gossip-3', 'gossip-8']
Cluster 3: ['gossip-5', 'gossip-9', 'gossip-1', 'gossip-6']

```