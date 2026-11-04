"""Read-only physical contact-chain scoring for several parts in one bin."""
from collections import deque


def support_chain(target, bin_geoms, part_geoms, contact_pairs):
    """Rigid-part nodes connect all their geoms; floating cycles have no root."""
    root = 'container'
    owners = {int(g): part for part, geoms in part_geoms.items() for g in geoms}
    owners.update({int(g): root for g in bin_geoms})
    adjacent = {root: set(), **{part: set() for part in part_geoms}}
    for a, b in contact_pairs:
        left, right = owners.get(int(a)), owners.get(int(b))
        if left is None or right is None or left == right:
            continue
        adjacent[left].add(right); adjacent[right].add(left)
    queue = deque([(target, [target])]); visited = {target}
    while queue:
        node, path = queue.popleft()
        if node == root:
            return path
        for other in sorted(adjacent.get(node, ()), key=str):
            if other not in visited:
                visited.add(other); queue.append((other, path+[other]))
    return None
