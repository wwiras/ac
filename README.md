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
