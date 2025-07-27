# Agglomerative Clustering
Agglomerative Clustering with BNSF

### Step 1 - Create a network topology (based on ER or BA)
Below is the example on how to create a network overlay of BA/ER network.

```shell
# Create overlay networkDeployments. Change the values accordingly
$ python network_constructor.py --nodes 10 --model BA --minlat 10 --maxlat 98 --others 5
minlat: 10
maxlat: 98
Initial status from the input .....
Number of nodes in the network: 10
Average neighbor (degree): 5
Creating BARABASI ALBERT (BA) network model .....
Average degree: 5.0
Target degree:5
nx.is_connected(network): True
Graph Before: Graph with 10 nodes and 25 edges
BA network model is SUCCESSFUL ! ....
Graph After: Graph with 10 nodes and 25 edges
Do you want to save the graph? (y/n): y 
Topology saved to nodes10_Jul262025125801_BA5.json

```

### Step 2 - Apply Agglomerative Clustering (AC)
The input are BA topology (json) and M total clusters. Once AC processed completed, a new overlay network
is generated in json file. This new file name is formatted by combining its original filename plus "AC"
in the end.
```shell
(venv) $ python AC.py --json_file_path nodes10_Jul262025125801_BA5.json --num_clusters 3
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


--- BNSF Process Completed ---

--- Final Clusters (M = 3) ---
Cluster 1: ['gossip-2', 'gossip-4', 'gossip-0', 'gossip-7']
Cluster 2: ['gossip-3', 'gossip-8']
Cluster 3: ['gossip-5', 'gossip-9', 'gossip-1', 'gossip-6']

--- Leader Selection & Announcement ---
Cluster 1 Leader: gossip-7
Cluster 2 Leader: gossip-8
Cluster 3 Leader: gossip-9

--- Phase 3: MST Construction (Parallel Logic) ---
  Cluster 1 MST Edges (Root: 2):
    Edge: 0 -> 4, Weight: 14.0
    Edge: 2 -> 0, Weight: 45.0
    Edge: 0 -> 7, Weight: 10.0
  Cluster 2 MST Edges (Root: 3):
    Edge: 3 -> 8, Weight: 59.0
  Cluster 3 MST Edges (Root: 5):
    Edge: 6 -> 9, Weight: 35.0
    Edge: 6 -> 1, Weight: 19.0
    Edge: 5 -> 6, Weight: 60.0

--- Connecting MST Root Nodes ---
  Inter-cluster connecting edges:
    Edge: 5 -> 3, Weight: 35.0
    Edge: 2 -> 5, Weight: 23.0

--- Comprehensive MST for the Entire Network ---
  Edge 1: gossip-5 -> gossip-3, Weight: 35.0
  Edge 2: gossip-2 -> gossip-5, Weight: 23.0
  Edge 3: gossip-0 -> gossip-4, Weight: 14.0
  Edge 4: gossip-2 -> gossip-0, Weight: 45.0
  Edge 5: gossip-0 -> gossip-7, Weight: 10.0
  Edge 6: gossip-3 -> gossip-8, Weight: 59.0
  Edge 7: gossip-6 -> gossip-9, Weight: 35.0
  Edge 8: gossip-6 -> gossip-1, Weight: 19.0
  Edge 9: gossip-5 -> gossip-6, Weight: 60.0

--- Phase 4: Neighbor Selection (MON) ---
  Node gossip-0's Optimal Neighbors:
    - gossip-4 (Weight: 14.0)
    - gossip-2 (Weight: 45.0)
    - gossip-7 (Weight: 10.0)
  Node gossip-1's Optimal Neighbors:
    - gossip-6 (Weight: 19.0)
  Node gossip-2's Optimal Neighbors:
    - gossip-5 (Weight: 23.0)
    - gossip-0 (Weight: 45.0)
  Node gossip-3's Optimal Neighbors:
    - gossip-5 (Weight: 35.0)
    - gossip-8 (Weight: 59.0)
  Node gossip-4's Optimal Neighbors:
    - gossip-0 (Weight: 14.0)
  Node gossip-5's Optimal Neighbors:
    - gossip-3 (Weight: 35.0)
    - gossip-2 (Weight: 23.0)
    - gossip-6 (Weight: 60.0)
  Node gossip-6's Optimal Neighbors:
    - gossip-9 (Weight: 35.0)
    - gossip-1 (Weight: 19.0)
    - gossip-5 (Weight: 60.0)
  Node gossip-7's Optimal Neighbors:
    - gossip-0 (Weight: 10.0)
  Node gossip-8's Optimal Neighbors:
    - gossip-3 (Weight: 59.0)
  Node gossip-9's Optimal Neighbors:
    - gossip-6 (Weight: 35.0)

--- Phase 5: Building New Overlay Topology File ---
Successfully created new overlay topology file at: topology/nodes10_Jul262025125801_BA5_AC.json

```

### Step 3 - Test Newly Generated Agglomerative Clustering (AC) in the simulator
The input are BA topology (json) and M total clusters. Once AC processed completed, a new overlay network
is generated in json file. This new file name is formatted by combining its original filename plus "AC"
in the end. In the example above, the new json file (overlay network) is "nodes10_Jul262025125801_BA5_AC.json" .
```shell