from typing import List, Tuple, Any
from tabulate import tabulate

# Dijkstra class
class Dijkstra():

    # Constructor to initiliaze dijkstra
    def __init__(self, nodes: List[str], adj_mat: List[int], src: str):
        self.nodes = nodes
        self.adj_mat = adj_mat
        self.src = src
        self.queue = [[float("inf"), None] for _ in range(len(nodes))]
        src_index = nodes.index(src)
        self.queue[src_index] = [0, src_index]
        self.visited = [False for _ in range(len(nodes))]

    # Get the index of an unvisited node with the minimum distance from source
    def find_min_node_index(self) -> int:
        min_index = -1
        for i, v_node in enumerate(self.visited):
            if not v_node and (min_index == -1 or self.queue[i][0] < self.queue[min_index][0]):
                min_index = i
        return min_index

    # Executes dijkstra
    def execute(self) -> List[List[Any]]:
        # Initialize history
        history = []
        # Index of minimum distance node
        min_index = self.find_min_node_index()
        nodes_len = len(self.nodes)
        while min_index >= 0:
            # Mark min_index as visited
            self.visited[min_index] = True

            # Iterate through neighbors
            for i in range(min_index*nodes_len, min_index*nodes_len + nodes_len):
                neighbor_dist = self.adj_mat[i]
                # In adjacent matrix, neighbor_dist(weight)=0 means it's not a neigbor
                if neighbor_dist > 0:
                    dist = self.queue[min_index][0] + neighbor_dist
                    # Update distance from source if less
                    if dist < self.queue[i % nodes_len][0]:
                        self.queue[i % nodes_len] = [dist, min_index]

            # Save state to history
            history.append([min_index, self.queue.copy()])

            # Get the new index of minimum distance node
            min_index = self.find_min_node_index()

        return history

    # Builds path from source to destination
    def construct_path(self, dest: str) -> Tuple[List[int], List[int]]:
        src_index = self.nodes.index(src)
        dest_index = self.nodes.index(dest)
        nodes_len = len(self.nodes)
        path = []
        path_dist = []
        # Insert path destination by node index
        path.insert(0, dest_index)
        prev_node_index = self.queue[dest_index][1]
        # If there's no previous node on destination then there's no path from source
        if prev_node_index is None:
            return None
        # Insert path distance
        path_dist.insert(0, self.adj_mat[prev_node_index*nodes_len + dest_index])
        while prev_node_index is not src_index:
            temp = prev_node_index
            # Insert path by node index
            path.insert(0, prev_node_index)
            prev_node_index = self.queue[prev_node_index][1]
            # Insert path distance
            path_dist.insert(0, self.adj_mat[prev_node_index*nodes_len + temp])
        # Insert path source by node index
        path.insert(0, src_index)
        # Insert path distance
        path_dist.insert(0, self.adj_mat[prev_node_index*nodes_len + src_index])
        return (path, path_dist)


# Main entry
if __name__ == "__main__":
    # Initialize nodes
    nodes = ["Monaire", "Poirott", "Milis", "Bouche", "Tempest", "Ranoa", "Jura", "Asura"]
    # Initialize adjacent matrix
    adj_mat = [
        0, 4, 13, 0, 0, 0, 0, 0,
        0, 0, 5, 8, 0, 0, 0, 0,
        0, 0, 0, 0, 5, 10, 0, 0,
        0, 0, 0, 0, 3, 0, 3, 14,
        0, 0, 0, 0, 0, 6, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 5,
        0, 0, 0, 0, 0, 0, 0, 12,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    # Initialize source
    src = "Monaire"
    # Initialize destination
    dest = "Asura"
    # Initialize Dijkstra class
    newDijkstra = Dijkstra(nodes, adj_mat, src)
    # History table for table printing
    history_table = []
    # Execute dijkstra and get history
    history = newDijkstra.execute()
    # Format history to history_table
    for history_item in history:
        queue = history_item[1]
        history_row = [nodes[history_item[0]]]
        for queue_item in queue:
            history_cell = [str(queue_item[0])]
            if queue_item[1] is not None:
                history_cell.append(nodes[queue_item[1]])
            history_row.append("_".join(history_cell))
        history_table.append(history_row)

    # Print history_table
    print(tabulate(history_table, headers=["V", *nodes], tablefmt="simple_grid"))

    # Construct path to destination
    path = newDijkstra.construct_path(dest)
    if path is None:
        # Tell the user if there's no path to get to the destination
        print(f"There exists no path from {src} to {dest}")
    else:
        # Print the shortest path
        print(f"Shortest path from {src} to {dest}:")
        print(" + ".join([str(dist) for i, dist in enumerate(path[1]) if i > 0]) + " = " + str(sum(path[1])))
        print(" -> ".join([nodes[node_i] for node_i in path[0]]))
