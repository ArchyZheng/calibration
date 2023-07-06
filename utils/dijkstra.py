# %%
import numpy as np
from collections import deque
from collections import defaultdict
import operator
import heapq


def shortest_path(img_data: np.array, begin: [int, int], end: [int, int]) -> np.array:
    """
    return the path recoder, which recode the locations(x, y) from begin point towards the end point in order.
    :param img_data:
    :param begin: the top of the image
    :param end: the bottom of the image
    """
    width, height = img_data.shape[0], img_data.shape[1]
    seen = defaultdict(lambda: float('inf'))
    seen[begin] = 0

    previous = {begin: None}

    def get_cost(source_point: (int, int), around_point: (int, int)):
        return 510. - img_data[source_point] - img_data[around_point] + seen[source_point]

    def available(point: (int, int)):
        current_x = point[0]
        current_y = point[1]
        return 0 <= current_x < width and 0 <= current_y < height and mark_matrix[point] != 1

    direction = ([0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1])

    mark_matrix = np.zeros_like(img_data)

    queue = deque()
    queue.append(begin)
    while queue:
        source_point = queue.popleft()
        mark_matrix[source_point] = 1

        for bias in direction:
            around_point = tuple(map(lambda a, b: a + b, source_point, bias))
            if available(around_point) and around_point not in queue:
                queue.append(around_point)
                attempt_cost = get_cost(source_point=source_point, around_point=around_point)
                if seen[around_point] > attempt_cost:
                    seen[around_point] = attempt_cost
                    previous[around_point] = source_point
        queue = sorted(queue, key=lambda x:seen[x])
        queue = deque(queue)

    path = deque()
    current = end
    while current:
        path.append(current)
        current = previous[current]

    return path, seen
