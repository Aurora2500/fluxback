
def get_edge_set(start):
	edges = set()
	stack = [start]
	visited = set()
	while len(stack) != 0:
		node = stack.pop()
		if node.dependencies is None:
			continue
		for dependency in node.dependencies:
			edges.add((node, dependency))
			if dependency not in visited:
				stack.append(dependency)
				visited.add(dependency)
	return edges



def topology_sort(out_node):
	edges = get_edge_set(out_node)
	free_set = {out_node}
	sorted_list = []
	while len(free_set) != 0:
		node = free_set.pop();
		sorted_list.append(node)
		if not node.dependencies:
			continue
		for dependency in node.dependencies:
			edges.discard((node, dependency))
			for free in free_set:
				if (free, dependency) in edges:
					break
			else:
				free_set.add(dependency)
	if len(edges) != 0:
		raise "Cyclic graph"
	return sorted_list