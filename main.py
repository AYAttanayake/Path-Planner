import heapq
import matplotlib.pyplot as plt
import numpy as np

# Square grids up to 100x100
grid = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

start = (5, 6)
dst = (9, 9)
grid[start] = 0
grid[dst] = 0

# Eucledian distance
def heuristic(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

# A* Algorithm
def astar(grid, start, dst):
    open_set = []
    heapq.heappush(open_set, (0, start))

    last_visit = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, dst)}

    directions = [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
        (-1, -1, 1.41), (-1, 1, 1.41), (1, -1, 1.41), (1, 1, 1.41)
    ]

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == dst:
            path = []
            while current in last_visit:
                path.append(current)
                current = last_visit[current]
            path.append(start)
            return path[::-1]

        x, y = current
        for dx, dy, cost in directions:
            neighbor = (x + dx, y + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor] == 1:
                    continue
                # Avoid diagonal movement between diagonally adjacent occupied cells
                if abs(dx) == 1 and abs(dy) == 1:
                    if grid[x][y + dy] == 1 or grid[x + dx][y] == 1:
                        continue
                predicted_g = g_score[current] + cost
                if predicted_g < g_score.get(neighbor, float('inf')):
                    last_visit[neighbor] = current
                    g_score[neighbor] = predicted_g
                    f_score[neighbor] = predicted_g + heuristic(neighbor, dst) * (1.0 + 1e-5)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

path = astar(grid, start, dst)

# Visualization
plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='gray_r')

if path:
    path_y, path_x = zip(*path)
    plt.plot(path_x, path_y, color='red', linewidth=2)

plt.scatter(*start[::-1], color='green', label='Start')
plt.scatter(*dst[::-1], color='red', label='Destination')
plt.legend()
plt.title("Calculated path")
plt.show()
