import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import math
import random
from scipy.spatial import ConvexHull
from collections import deque
import heapq

class Robot:
    def __init__(self, x, y, radius=0.5, lidar_range=5.0, lidar_resolution=360):
        self.x = x
        self.y = y
        self.radius = radius
        self.lidar_range = lidar_range
        self.lidar_resolution = lidar_resolution
        self.lidar_angles = np.linspace(-np.pi, np.pi, lidar_resolution, endpoint=True)  # 360-degree view
        self.lidar_data = np.ones(lidar_resolution) * lidar_range
        self.velocity = 0.5  # Base velocity
        self.max_velocity = 1.0  # Maximum velocity
        self.path_history = [(x, y)]
        self.orientation = 0  # Robot's orientation in radians
        self.lidar_points = []  # Store LIDAR end points for visualization
        self.current_path = []  # Current planned path
        
    def check_collision_with_obstacles(self, x, y, obstacles):
        """Check if a point collides with any obstacle using true O(n) implementation"""
        # Convert query point to numpy array for vectorized operations
        point = np.array([x, y])
        
        # Process obstacles in batches for better vectorization
        batch_size = 32
        for i in range(0, len(obstacles), batch_size):
            batch = obstacles[i:i+batch_size]
            
            # Pre-compute all centers and bounding boxes in one go
            centers = np.array([np.mean(obs.get_vertices(), axis=0) for obs in batch])
            
            # Vectorized center distance check
            dists_to_centers = np.linalg.norm(centers - point, axis=1)
            close_centers = dists_to_centers < (self.radius + 2.0)
            
            if np.any(close_centers):
                return True, batch[np.argmax(close_centers)]
            
            # Only check vertices for obstacles that might be close
            potential_collisions = dists_to_centers < (self.radius + 4.0)
            if not np.any(potential_collisions):
                continue
                
            # Check vertices only for potentially colliding obstacles
            for idx, obstacle in enumerate(batch):
                if potential_collisions[idx]:
                    vertices = np.array(obstacle.get_vertices())
                    dists = np.linalg.norm(vertices - point, axis=1)
                    if np.any(dists < (self.radius + 1.5)):
                        return True, obstacle
        
        return False, None

    def find_path_to_goal(self, goal_x, goal_y, obstacles):
        """Find a path to the goal using O(n) Bug algorithm with dynamic obstacle avoidance"""
        # Convert to numpy arrays for vectorized operations
        start = np.array([self.x, self.y])
        goal = np.array([goal_x, goal_y])
        
        # If direct path is clear, return it
        if not self.check_collision_with_obstacles(goal_x, goal_y, obstacles):
            return [(self.x, self.y), (goal_x, goal_y)]
        
        # Initialize path with current position
        path = [(self.x, self.y)]
        current_pos = start.copy()
        
        # Pre-compute obstacle centers and radii for O(1) access
        obstacle_centers = np.array([np.mean(obs.get_vertices(), axis=0) for obs in obstacles])
        obstacle_radii = np.array([np.max(np.linalg.norm(obs.get_vertices() - center, axis=1)) 
                                  for obs, center in zip(obstacles, obstacle_centers)])
        
        # Main loop - O(n) where n is number of obstacles
        while np.linalg.norm(current_pos - goal) > self.radius:
            # Find nearest obstacle in the direction of goal - O(n)
            direction = (goal - current_pos) / np.linalg.norm(goal - current_pos)
            obstacle_distances = np.linalg.norm(obstacle_centers - current_pos, axis=1)
            obstacle_angles = np.arccos(np.dot((obstacle_centers - current_pos) / 
                                      obstacle_distances[:, np.newaxis], direction))
            
            # Filter obstacles in front (within 90 degrees of goal direction)
            valid_obstacles = obstacle_angles < np.pi/2
            if not np.any(valid_obstacles):
                # No obstacles between current position and goal
                path.append((goal_x, goal_y))
                break
            
            # Get nearest obstacle
            nearest_idx = np.argmin(obstacle_distances[valid_obstacles])
            nearest_obstacle = obstacles[nearest_idx]
            
            # Follow obstacle boundary - O(1) per vertex
            vertices = nearest_obstacle.get_vertices()
            boundary_point = vertices[np.argmin(np.linalg.norm(vertices - current_pos, axis=1))]
            
            # Add boundary point to path
            path.append((boundary_point[0], boundary_point[1]))
            current_pos = boundary_point
            
            # Try to move toward goal from new position
            if not self.check_collision_with_obstacles(goal_x, goal_y, obstacles):
                path.append((goal_x, goal_y))
                break
        
        return path
    
    def find_safe_direction(self, dx, dy, obstacles):
        """Find safe direction to move based on current path and obstacles"""
        if not self.current_path:
            return 0, 0
            
        # Get next waypoint
        target = self.current_path[0]
        dx = target[0] - self.x
        dy = target[1] - self.y
        
        # If close to waypoint, remove it
        if math.sqrt(dx*dx + dy*dy) < self.radius:
            self.current_path.pop(0)
            if self.current_path:
                target = self.current_path[0]
                dx = target[0] - self.x
                dy = target[1] - self.y
        
        # Normalize direction
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx = dx / dist * self.velocity
            dy = dy / dist * self.velocity
        
        # Check if movement is safe
        test_x = self.x + dx
        test_y = self.y + dy
        collision, _ = self.check_collision_with_obstacles(test_x, test_y, obstacles)
        if not collision:
            return dx, dy
        
        # If collision detected, try to avoid obstacle
        angles = np.linspace(-np.pi/4, np.pi/4, 8)
        for angle in angles:
            rot_dx = dx*math.cos(angle) - dy*math.sin(angle)
            rot_dy = dx*math.sin(angle) + dy*math.cos(angle)
            test_x = self.x + rot_dx
            test_y = self.y + rot_dy
            collision, _ = self.check_collision_with_obstacles(test_x, test_y, obstacles)
            if not collision:
                return rot_dx, rot_dy
        
        return 0, 0

    def move(self, goal_x, goal_y, obstacles):
        """Move robot towards goal with intelligent path planning"""
        # Get distance to goal
        dx = goal_x - self.x
        dy = goal_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Scale velocity based on distance to goal
        scale = min(1.0, dist / 5.0)  # Slow down when close to goal
        velocity = self.velocity * scale
        
        # Normalize direction
        if dist > 0:
            dx = dx / dist * velocity
            dy = dy / dist * velocity
        
        # Try moving directly towards goal first
        test_x = self.x + dx
        test_y = self.y + dy
        collision, _ = self.check_collision_with_obstacles(test_x, test_y, obstacles)
        
        if not collision:
            # Direct path is clear, move towards goal
            self.x = test_x
            self.y = test_y
        else:
            # Try to find alternative direction
            angles = np.linspace(-np.pi/2, np.pi/2, 8)  # Try 8 different angles
            for angle in angles:
                # Rotate the movement vector
                rot_dx = dx*math.cos(angle) - dy*math.sin(angle)
                rot_dy = dx*math.sin(angle) + dy*math.cos(angle)
                
                # Test new position
                test_x = self.x + rot_dx
                test_y = self.y + rot_dy
                collision, _ = self.check_collision_with_obstacles(test_x, test_y, obstacles)
                
                if not collision:
                    # Found safe direction, move there
                    self.x = test_x
                    self.y = test_y
                    break
        
        # Update path history
        self.path_history.append((self.x, self.y))
        
        # Update orientation based on movement direction
        if dx != 0 or dy != 0:
            self.orientation = math.atan2(dy, dx)
        # Update path if needed
        if not self.current_path:
            self.current_path = self.find_path_to_goal(goal_x, goal_y, obstacles)
            if not self.current_path:
                return  # No valid path found
        
        # Get movement direction
        dx, dy = self.find_safe_direction(goal_x - self.x, goal_y - self.y, obstacles)
        
        # Adjust velocity based on distance to goal and obstacles
        dist_to_goal = math.sqrt((goal_x - self.x)**2 + (goal_y - self.y)**2)
        self.velocity = min(self.max_velocity, max(0.2, dist_to_goal / 5.0))
        
        # Update position and orientation
        self.x += dx
        self.y += dy
        if dx != 0 or dy != 0:
            self.orientation = math.atan2(dy, dx)
        self.path_history.append((self.x, self.y))
        
    def scan_environment(self, obstacles, weighted_regions):
        # Reset LIDAR data
        self.lidar_data = np.ones(self.lidar_resolution) * self.lidar_range
        self.lidar_points = []
        
        # Scan for obstacles
        for obstacle in obstacles:
            vertices = obstacle.get_vertices()
            for i, rel_angle in enumerate(self.lidar_angles):
                # Calculate absolute angle based on robot's orientation
                abs_angle = rel_angle + self.orientation
                dx = math.cos(abs_angle)
                dy = math.sin(abs_angle)
                
                # Ray casting
                intersection = self.ray_intersect_polygon(self.x, self.y, dx, dy, vertices)
                if intersection:
                    dist = math.sqrt((intersection[0] - self.x)**2 + (intersection[1] - self.y)**2)
                    if dist < self.lidar_data[i]:
                        self.lidar_data[i] = dist
                        self.lidar_points.append(intersection)
                else:
                    # Add endpoint of maximum range if no intersection
                    end_x = self.x + dx * self.lidar_range
                    end_y = self.y + dy * self.lidar_range
                    self.lidar_points.append((end_x, end_y))
        
        return self.lidar_data
    
    def ray_intersect_polygon(self, x, y, dx, dy, vertices):
        min_dist = float('inf')
        closest_point = None
        
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i+1) % len(vertices)]
            
            # Line segment intersection algorithm
            x3, y3 = x, y
            x4, y4 = x + dx * self.lidar_range, y + dy * self.lidar_range
            
            # Calculate determinants
            den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if den == 0:
                continue  # Parallel lines
                
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
            
            # Check if intersection is within segment and ray
            if 0 <= ua <= 1 and ub >= 0:
                x_intersect = x1 + ua * (x2 - x1)
                y_intersect = y1 + ua * (y2 - y1)
                dist = math.sqrt((x_intersect - x)**2 + (y_intersect - y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (x_intersect, y_intersect)
        
        return closest_point

class Obstacle:
    def __init__(self, vertices, velocity=(0, 0), bounds=None):
        self.vertices = np.array(vertices, dtype=np.float64)
        self.original_vertices = np.array(vertices, dtype=np.float64)
        self.center = np.mean(vertices, axis=0)
        self.velocity = velocity
        self.bounds = bounds  # (x_min, y_min, x_max, y_max)
        self.trajectory = deque(maxlen=100)
        self.trajectory.append((self.center[0], self.center[1]))
        self.type = "static" if velocity == (0, 0) else \
                   ("predictable" if random.random() > 0.3 else "stochastic")
        # Add safety margin for collision detection
        self.safety_margin = 0.5
        # Add movement zone bounds
        if bounds:
            x_min, y_min, x_max, y_max = bounds
            width = x_max - x_min
            height = y_max - y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            # Assign each obstacle its own zone
            zone_width = width / 3
            zone_height = height / 3
            self.zone = (
                center_x - zone_width/2,
                center_y - zone_height/2,
                center_x + zone_width/2,
                center_y + zone_height/2
            )
    
    def update(self, dt=0.1, other_obstacles=None):
        # Always update position based on velocity
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        
        # Update vertices
        new_vertices = self.vertices + np.array([dx, dy])
        
        # Check bounds first
        if self.bounds:
            x_min, y_min, x_max, y_max = self.bounds
            # If hitting x bounds, reverse x velocity
            if np.min(new_vertices[:, 0]) < x_min or np.max(new_vertices[:, 0]) > x_max:
                self.velocity = (-self.velocity[0], self.velocity[1])
                dx = self.velocity[0] * dt  # Recalculate dx with new velocity
                new_vertices = self.vertices + np.array([dx, dy])
            
            # If hitting y bounds, reverse y velocity
            if np.min(new_vertices[:, 1]) < y_min or np.max(new_vertices[:, 1]) > y_max:
                self.velocity = (self.velocity[0], -self.velocity[1])
                dy = self.velocity[1] * dt  # Recalculate dy with new velocity
                new_vertices = self.vertices + np.array([dx, dy])
        
        # Check collision with other obstacles
        if other_obstacles:
            will_collide = False
            for other in other_obstacles:
                if other != self:
                    # Calculate future positions
                    other_future = other.vertices + np.array([other.velocity[0] * dt, other.velocity[1] * dt])
                    # Check if paths will cross
                    min_x1, max_x1 = np.min(new_vertices[:, 0]), np.max(new_vertices[:, 0])
                    min_y1, max_y1 = np.min(new_vertices[:, 1]), np.max(new_vertices[:, 1])
                    min_x2, max_x2 = np.min(other_future[:, 0]), np.max(other_future[:, 0])
                    min_y2, max_y2 = np.min(other_future[:, 1]), np.max(other_future[:, 1])
                    
                    if not (max_x1 < min_x2 or min_x1 > max_x2 or
                            max_y1 < min_y2 or min_y1 > max_y2):
                        will_collide = True
                        break
            
            if will_collide:
                # Simply reverse direction on collision
                self.velocity = (-self.velocity[0], -self.velocity[1])
                return
        
        # Update position
        self.vertices = new_vertices
        self.center = np.mean(self.vertices, axis=0)
        self.trajectory.append((self.center[0], self.center[1]))
        
        # Add slight randomness for stochastic obstacles
        if self.type == "stochastic" and random.random() < 0.02:  # Reduced randomness
            speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            angle = math.atan2(self.velocity[1], self.velocity[0])
            angle += random.uniform(-np.pi/12, np.pi/12)  # Small angle change
            self.velocity = (speed * math.cos(angle), speed * math.sin(angle))
    
    def get_vertices(self):
        return self.vertices
        
    def check_collision(self, other_obstacle):
        # Add safety margin to bounding boxes
        min_x1 = np.min(self.vertices[:, 0]) - self.safety_margin
        max_x1 = np.max(self.vertices[:, 0]) + self.safety_margin
        min_y1 = np.min(self.vertices[:, 1]) - self.safety_margin
        max_y1 = np.max(self.vertices[:, 1]) + self.safety_margin
        
        min_x2 = np.min(other_obstacle.vertices[:, 0]) - other_obstacle.safety_margin
        max_x2 = np.max(other_obstacle.vertices[:, 0]) + other_obstacle.safety_margin
        min_y2 = np.min(other_obstacle.vertices[:, 1]) - other_obstacle.safety_margin
        max_y2 = np.max(other_obstacle.vertices[:, 1]) + other_obstacle.safety_margin
        
        return not (max_x1 < min_x2 or min_x1 > max_x2 or
                   max_y1 < min_y2 or min_y1 > max_y2).tolist()
    
    def check_collision(self, other_obstacle):
        # Simple bounding box collision check
        self_min_x = np.min(self.vertices[:, 0])
        self_max_x = np.max(self.vertices[:, 0])
        self_min_y = np.min(self.vertices[:, 1])
        self_max_y = np.max(self.vertices[:, 1])
        
        other_min_x = np.min(other_obstacle.vertices[:, 0])
        other_max_x = np.max(other_obstacle.vertices[:, 0])
        other_min_y = np.min(other_obstacle.vertices[:, 1])
        other_max_y = np.max(other_obstacle.vertices[:, 1])
        
        # Check if bounding boxes overlap
        return not (self_max_x < other_min_x or self_min_x > other_max_x or 
                   self_max_y < other_min_y or self_min_y > other_max_y)

class WeightedRegion:
    def __init__(self, vertices, alpha):
        self.vertices = np.array(vertices)
        self.alpha = alpha
    
    def contains_point(self, x, y):
        # Ray casting algorithm to check if point is inside polygon
        inside = False
        j = len(self.vertices) - 1
        
        for i in range(len(self.vertices)):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside
            j = i
            
        return inside

class HDPredictionModel:
    def __init__(self):
        self.learning_rate = 0.01
        self.prediction_horizon = 10  # Number of steps ahead to predict
        
    def predict_obstacle_motion(self, obstacles, dt=0.1):
        predictions = []
        
        for obstacle in obstacles:
            if obstacle.type == "static":
                # Static obstacles don't move
                future_positions = [obstacle.center.copy() for _ in range(self.prediction_horizon)]
            elif obstacle.type == "predictable":
                # For predictable obstacles, simple linear extrapolation
                future_positions = []
                current_pos = obstacle.center.copy()
                vx, vy = obstacle.velocity
                
                for i in range(self.prediction_horizon):
                    next_pos = current_pos.copy()
                    next_pos[0] += vx * dt * (i+1)
                    next_pos[1] += vy * dt * (i+1)
                    future_positions.append(next_pos)
            else:  # "stochastic"
                # For stochastic obstacles, add uncertainty to predictions
                future_positions = []
                current_pos = obstacle.center.copy()
                vx, vy = obstacle.velocity
                
                for i in range(self.prediction_horizon):
                    noise_x = (random.random() - 0.5) * 0.1 * (i+1)
                    noise_y = (random.random() - 0.5) * 0.1 * (i+1)
                    next_pos = current_pos.copy()
                    next_pos[0] += vx * dt * (i+1) + noise_x
                    next_pos[1] += vy * dt * (i+1) + noise_y
                    future_positions.append(next_pos)
            
            predictions.append(future_positions)
        
        return predictions

class HDPathPlanner:
    def __init__(self, grid_size=0.5, grid_width=40, grid_height=40):
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = np.zeros((grid_height, grid_width))  # 0: free, 1: obstacle
        self.alpha = 0.5  # Exploration rate
        self.beta = 0.5   # Risk factor
        self.explored_areas = set()
        
    def update_grid(self, obstacles, weighted_regions):
        # Reset grid
        self.grid = np.zeros((self.grid_height, self.grid_width))
        
        # Mark obstacles
        for obstacle in obstacles:
            vertices = obstacle.get_vertices()
            min_x = min(v[0] for v in vertices)
            max_x = max(v[0] for v in vertices)
            min_y = min(v[1] for v in vertices)
            max_y = max(v[1] for v in vertices)
            
            # Convert to grid coordinates
            min_i = max(0, int(min_y / self.grid_size))
            max_i = min(self.grid_height-1, int(max_y / self.grid_size))
            min_j = max(0, int(min_x / self.grid_size))
            max_j = min(self.grid_width-1, int(max_x / self.grid_size))
            
            # Mark cells inside obstacle
            for i in range(min_i, max_i+1):
                for j in range(min_j, max_j+1):
                    # Check if cell center is inside obstacle
                    cell_y = (i + 0.5) * self.grid_size
                    cell_x = (j + 0.5) * self.grid_size
                    
                    for weighted_region in weighted_regions:
                        if weighted_region.contains_point(cell_x, cell_y):
                            self.grid[i, j] = weighted_region.alpha
            
            # Mark obstacle cells with 1 (blocked)
            for i in range(min_i, max_i+1):
                for j in range(min_j, max_j+1):
                    # Simplified check: just mark the bounding box
                    self.grid[i, j] = 1
    
    def compute_global_path(self, start, goal, obstacles):
        # Convert to grid coordinates
        start_i = int(start[1] / self.grid_size)
        start_j = int(start[0] / self.grid_size)
        goal_i = int(goal[1] / self.grid_size)
        goal_j = int(goal[0] / self.grid_size)
        
        # A* algorithm
        open_set = [(0, (start_i, start_j))]  # Priority queue: (f_score, position)
        came_from = {}
        
        g_score = {(start_i, start_j): 0}
        f_score = {(start_i, start_j): self.heuristic((start_i, start_j), (goal_i, goal_j))}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == (goal_i, goal_j):
                # Reconstruct path
                path = []
                while current in came_from:
                    i, j = current
                    path.append((j * self.grid_size, i * self.grid_size))
                    current = came_from[current]
                path.append((start[0], start[1]))
                path.reverse()
                return path
            
            i, j = current
            # Check all 8 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_height and 0 <= nj < self.grid_width:
                        # Skip obstacle cells
                        if self.grid[ni, nj] == 1:
                            continue
                        
                        # Calculate new g_score
                        move_cost = math.sqrt(di**2 + dj**2)
                        # Apply weighted region cost if applicable
                        if self.grid[ni, nj] > 0 and self.grid[ni, nj] < 1:
                            move_cost *= (1 + self.grid[ni, nj])
                            
                        tentative_g = g_score[current] + move_cost
                        
                        if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                            came_from[(ni, nj)] = current
                            g_score[(ni, nj)] = tentative_g
                            f = tentative_g + self.heuristic((ni, nj), (goal_i, goal_j))
                            f_score[(ni, nj)] = f
                            heapq.heappush(open_set, (f, (ni, nj)))
                            
                            # Mark as explored
                            self.explored_areas.add((ni, nj))
        
        # No path found
        return None
    
    def heuristic(self, a, b):
        # Euclidean distance
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def compute_local_path(self, current_pos, global_path, lidar_data, lidar_angles):
        # Find the point on the global path to aim for
        # This is a simplified version that just aims for the next point on the global path
        if len(global_path) <= 1:
            return current_pos  # Already at goal
        
        target_point = global_path[1]  # Next point on global path
        
        # Check if obstacles are in the way using LIDAR data
        obstacles_in_way = False
        direction = math.atan2(target_point[1] - current_pos[1], target_point[0] - current_pos[0])
        
        # Find closest LIDAR angle to direction
        angle_diff = np.abs(np.array(lidar_angles) - direction)
        closest_idx = np.argmin(angle_diff)
        
        # If obstacle is too close, find alternative path
        if lidar_data[closest_idx] < 1.0:
            obstacles_in_way = True
        
        if obstacles_in_way:
            # Simple obstacle avoidance: find clearest direction
            clearest_idx = np.argmax(lidar_data)
            clearest_angle = lidar_angles[clearest_idx]
            
            # Move in clearest direction but try to stay close to global path
            # This is a simplified approach - a real implementation would be more sophisticated
            avoidance_distance = 0.5
            new_x = current_pos[0] + avoidance_distance * math.cos(clearest_angle)
            new_y = current_pos[1] + avoidance_distance * math.sin(clearest_angle)
            
            return (new_x, new_y)
        else:
            # No obstacles, follow global path
            return target_point
    
    def select_best_path(self, paths, current_pos, goal_pos, obstacles, alpha=0.5, beta=0.5):
        if not paths:
            return None
        
        best_score = float('inf')
        best_path = None
        
        for path in paths:
            # Calculate path cost
            distance_cost = 0
            risk_cost = 0
            
            for i in range(1, len(path)):
                # Distance cost
                segment_length = math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
                distance_cost += segment_length
                
                # Risk cost - distance to nearest obstacle
                min_obstacle_dist = float('inf')
                for obstacle in obstacles:
                    for vertex in obstacle.get_vertices():
                        dist = math.sqrt((path[i][0] - vertex[0])**2 + (path[i][1] - vertex[1])**2)
                        min_obstacle_dist = min(min_obstacle_dist, dist)
                
                # Inverse distance as risk (closer to obstacle = higher risk)
                if min_obstacle_dist > 0:
                    risk_cost += 1 / min_obstacle_dist
            
            # Total cost with alpha/beta weighting
            total_cost = alpha * distance_cost + beta * risk_cost
            
            if total_cost < best_score:
                best_score = total_cost
                best_path = path
        
        return best_path
    
    def update_parameters(self, obstacle_density):
        # Dynamically update alpha and beta based on obstacle density
        if obstacle_density > 0.3:
            # More dense environment, prioritize safety
            self.alpha = 0.3
            self.beta = 0.7
        else:
            # Less dense environment, prioritize efficiency
            self.alpha = 0.7
            self.beta = 0.3

class WeightedRegion:
    """Represents a region with a velocity multiplier and color"""
    def __init__(self, vertices, alpha, color='blue'):
        self.vertices = vertices  # List of (x,y) tuples defining the region
        self.alpha = alpha  # Velocity multiplier (0-1)
        self.color = color  # Color for visualization
        
    def point_inside(self, x, y):
        """Check if a point is inside the region using ray casting algorithm"""
        inside = False
        j = len(self.vertices) - 1
        for i in range(len(self.vertices)):
            if ((self.vertices[i][1] > y) != (self.vertices[j][1] > y) and
                x < (self.vertices[j][0] - self.vertices[i][0]) * (y - self.vertices[i][1]) /
                    (self.vertices[j][1] - self.vertices[i][1]) + self.vertices[i][0]):
                inside = not inside
            j = i
        return inside

class HDAlgorithm:
    def __init__(self, env_size=(20, 20)):
        self.env_size = env_size
        self.obstacles = []
        # Define weighted regions with 4 different types of terrain
        self.weighted_regions = [
            # Dense Forest (very slow) - Dark green
            WeightedRegion([(0, 13), (6, 13), (6, 19), (0, 19)], 0.5, 'darkgreen'),
            WeightedRegion([(14, 0), (20, 0), (20, 6), (14, 6)], 0.5, 'darkgreen'),
            
            # Rocky Terrain (slow) - Gray
            WeightedRegion([(7, 13), (13, 13), (13, 19), (7, 19)], 0.8, 'dimgray'),
            WeightedRegion([(0, 0), (6, 0), (6, 6), (0, 6)], 0.8, 'dimgray'),
            
            # Marsh/Swamp (medium speed) - Olive
            WeightedRegion([(14, 13), (20, 13), (20, 19), (14, 19)], 1.2, 'olive'),
            WeightedRegion([(7, 0), (13, 0), (13, 6), (7, 6)], 1.2, 'olive'),
            
            # Moon Surface/Road (fast) - Light gray
            WeightedRegion([(0, 7), (6, 7), (6, 12), (0, 12)], 1.8, 'lightgray'),
            WeightedRegion([(7, 7), (13, 7), (13, 12), (7, 12)], 1.8, 'lightgray'),
            WeightedRegion([(14, 7), (20, 7), (20, 12), (14, 12)], 1.8, 'lightgray'),
        ]
        self.robot = None
        self.goal = None
        self.goal_radius = 0.5  # Goal catch radius
        self.step_counter = 0
        self.goal_velocity = 1.0  # Increased goal movement speed
        self.goal_bounds = (14, 16, 19, 19)  # Wider movement bounds
        self.goal_direction = 1  # 1 for forward, -1 for backward

    def initialize(self):
        """Initialize the HD-BUG algorithm"""
        # Create robot as a small point
        self.robot = Robot(2.0, 2.0, radius=0.15, lidar_range=5.0, lidar_resolution=360)
        # Increase base velocity for faster movement
        self.robot.velocity = 1.0  # Significantly increased from default 0.5
        self.robot.max_velocity = 1.5  # Significantly increased from default 1.0
        
        # Create moving goal in upper right area
        self.goal = [18.0, 18.0]  # Make it a list for mutability
        self.goal_velocity = 0.25  # Moderate movement speed
        self.goal_direction = 1  # Start moving forward

        # Create obstacles
        self.create_obstacles()

        # Reset step counter
        self.step_counter = 0

        # Initialize reached_goal flag
        self.reached_goal = False

        print("HD-BUG Algorithm initialized")

    def create_obstacles(self):
        """Create moving obstacles in the environment"""
        # Create strategic obstacles with initial velocities
        obstacles_config = [
            {
                'vertices': [(5, 5), (7, 5), (7, 7), (5, 7)],  # Square
                'velocity': (0.8, 0.6),
                'bounds': (4, 4, 8, 8)
            },
            {
                'vertices': [(10, 8), (12, 8), (11, 10)],  # Triangle
                'velocity': (-0.7, 0.7),
                'bounds': (9, 7, 13, 11)
            },
            {
                'vertices': [(8, 12), (10, 12), (10, 14), (8, 14)],  # Rectangle
                'velocity': (0.6, -0.6),
                'bounds': (7, 11, 11, 15)
            },
            {
                'vertices': [(3, 10), (4, 10), (4, 13), (3, 13)],  # Vertical wall
                'velocity': (0.9, 0),
                'bounds': (2, 9, 5, 14)
            },
            {
                'vertices': [(14, 4), (15, 4), (16, 5), (15, 6), (14, 6)],  # Pentagon
                'velocity': (0.5, 0.8),
                'bounds': (13, 3, 17, 7)
            },
            {
                'vertices': [(2, 16), (3, 15), (4, 16), (4, 17), (3, 18), (2, 17)],  # Hexagon
                'velocity': (-0.6, -0.6),
                'bounds': (1, 14, 5, 19)
            },
            {
                'vertices': [(15, 12), (16, 11), (17, 11), (18, 12), (17, 13), (16, 13)],  # Star-like
                'velocity': (0.7, -0.5),
                'bounds': (14, 10, 19, 14)
            },
            {
                'vertices': [(7, 2), (8, 2), (9, 3), (9, 4), (8, 5), (7, 5), (6, 4), (6, 3)],  # Octagon
                'velocity': (-0.5, 0.5),
                'bounds': (5, 1, 10, 6)
            },
            # Add U-shaped dynamic obstacle
            {
                'vertices': [(12, 16), (12, 18), (13, 18), (13, 16.5), (15, 16.5), (15, 18), (16, 18), (16, 16)],  # U-shape
                'velocity': (0.3, -0.3),
                'bounds': (11, 15, 17, 19)
            }
        ]

        for config in obstacles_config:
            obstacle = Obstacle(
                vertices=config['vertices'],
                velocity=config['velocity'],
                bounds=config['bounds']
            )
            self.obstacles.append(obstacle)

    def create_weighted_regions(self):
        # Create two weighted regions with different alpha values
        region1 = WeightedRegion([
            (2, 2), (6, 2), (6, 6), (2, 6)
        ], alpha=0.3)

        region2 = WeightedRegion([
            (10, 10), (15, 10), (15, 15), (10, 15)
        ], alpha=0.5)

        self.weighted_regions = [region1, region2]

    def update_goal_position(self, dt=0.1):
        """Update goal position with back and forth movement"""
        x_min, y_min, x_max, y_max = self.goal_bounds

        # Update x position (moving horizontally)
        new_x = self.goal[0] + self.goal_velocity * self.goal_direction * dt

        # Check bounds and reverse direction if needed
        if new_x <= x_min:
            new_x = x_min
            self.goal_direction = 1  # Start moving forward
        elif new_x >= x_max:
            new_x = x_max
            self.goal_direction = -1  # Start moving backward

        # Update goal position
        self.goal[0] = new_x

    def check_goal_reached(self):
        """Check if the robot has reached the goal with a more generous radius"""
        dx = self.goal[0] - self.robot.x
        dy = self.goal[1] - self.robot.y
        distance = math.sqrt(dx**2 + dy**2)
        # More generous goal radius to make it easier to reach
        return distance < (self.goal_radius * 1.5)

    def step(self):
        """Execute one step of the algorithm"""
        # Update obstacles with collision checking
        for obstacle in self.obstacles:
            obstacle.update(other_obstacles=self.obstacles)

        # Update goal position
        self.update_goal_position()

        # Check if goal is reached
        if self.check_goal_reached():
            print(f"Goal reached in {self.step_counter} steps!")
            return True

        # Calculate distance to goal for logging
        dx = self.goal[0] - self.robot.x
        dy = self.goal[1] - self.robot.y
        distance = math.sqrt(dx**2 + dy**2)
        print(f"Step {self.step_counter + 1}: Distance to goal = {distance:.2f}")

        # Move robot towards goal while avoiding obstacles
        if distance > 0:
            # Scale velocity based on distance to goal and weighted regions
            scale = min(1.0, distance / 5.0)  # Slow down when close to goal

            # Check if robot is in any weighted region
            region_multiplier = 1.0
            for region in self.weighted_regions:
                if region.point_inside(self.robot.x, self.robot.y):
                    region_multiplier = region.alpha
                    break

            velocity = self.robot.velocity * scale * region_multiplier

            # Try different movement strategies in order of preference
            moved = False

            # Strategy 1: Direct path to goal
            if not moved:
                dx_norm = dx / distance * velocity
                dy_norm = dy / distance * velocity
                test_x = self.robot.x + dx_norm
                test_y = self.robot.y + dy_norm
                collision, _ = self.robot.check_collision_with_obstacles(test_x, test_y, self.obstacles)
                if not collision:
                    self.robot.x = test_x
                    self.robot.y = test_y
                    moved = True

            # Strategy 2: Try moving at various angles
            if not moved:
                angles = np.linspace(-np.pi, np.pi, 16)  # Try more angles
                for angle in angles:
                    rot_dx = velocity * math.cos(angle)
                    rot_dy = velocity * math.sin(angle)
                    test_x = self.robot.x + rot_dx
                    test_y = self.robot.y + rot_dy
                    collision, _ = self.robot.check_collision_with_obstacles(test_x, test_y, self.obstacles)
                    if not collision:
                        # Calculate how good this direction is (closer to goal is better)
                        new_dist = math.sqrt((test_x - self.goal[0])**2 + (test_y - self.goal[1])**2)
                        if new_dist < distance:  # Only move if it gets us closer to the goal
                            self.robot.x = test_x
                            self.robot.y = test_y
                            moved = True
                            break

            # Strategy 3: If stuck, try random movement
            if not moved and self.step_counter % 10 == 0:  # Only try random movement occasionally
                for _ in range(8):  # Try 8 random directions
                    angle = random.uniform(-np.pi, np.pi)
                    rot_dx = velocity * math.cos(angle)
                    rot_dy = velocity * math.sin(angle)
                    test_x = self.robot.x + rot_dx
                    test_y = self.robot.y + rot_dy
                    collision, _ = self.robot.check_collision_with_obstacles(test_x, test_y, self.obstacles)
                    if not collision:
                        self.robot.x = test_x
                        self.robot.y = test_y
                        moved = True
                        break

            if moved:
                # Update robot's orientation based on actual movement direction
                actual_dx = self.robot.x - self.robot.path_history[-1][0]
                actual_dy = self.robot.y - self.robot.path_history[-1][1]
                if actual_dx != 0 or actual_dy != 0:
                    self.robot.orientation = math.atan2(actual_dy, actual_dx)

                # Update path history
                self.robot.path_history.append((self.robot.x, self.robot.y))

        # Update step counter
        self.step_counter += 1
        return False

def main():
    # Create and initialize the algorithm
    hd = HDAlgorithm(env_size=(20, 20))
    hd.initialize()

    # Set up the figure and animation with higher DPI for better quality
    plt.rcParams['figure.dpi'] = 150
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, hd.env_size[0])
    ax.set_ylim(0, hd.env_size[1])

    # Add grid with units
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(np.arange(0, hd.env_size[0]+1, 2))
    ax.set_yticks(np.arange(0, hd.env_size[1]+1, 2))
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')

    # Plot robot as a small point
    robot_circle = plt.Circle((hd.robot.x, hd.robot.y), hd.robot.radius, 
                            facecolor='black',
                            edgecolor='white',
                            linewidth=0.8,
                            fill=True, 
                            label='Robot', 
                            zorder=6)

    # Plot goal as a cross
    cross_size = 0.5
    goal_marker = plt.Line2D([hd.goal[0]-cross_size, hd.goal[0]+cross_size], 
                           [hd.goal[1], hd.goal[1]], 
                           lw=2, color='red', zorder=6)
    ax.add_line(goal_marker)
    goal_marker2 = plt.Line2D([hd.goal[0], hd.goal[0]], 
                            [hd.goal[1]-cross_size, hd.goal[1]+cross_size], 
                            lw=2, color='red', zorder=6)
    ax.add_line(goal_marker2)

    # Create a circle around the cross for collision detection
    goal_circle = plt.Circle((hd.goal[0], hd.goal[1]), hd.goal_radius, 
                           facecolor='none',
                           edgecolor='red',
                           linewidth=1.0,
                           linestyle='--',
                           fill=False, 
                           label='Goal', 
                           zorder=6,
                           alpha=0.7)
    path_line, = ax.plot([], [], 'b-', alpha=0.7, linewidth=2, 
                        label='Robot Path', zorder=4)

    # Initialize LIDAR visualization with enhanced visibility
    lidar_lines = []
    for _ in range(hd.robot.lidar_resolution):
        line, = ax.plot([], [], 'yellow', alpha=0.3, linewidth=1.0, zorder=2,
                       solid_capstyle='round')
        lidar_lines.append(line)

    # Initialize weighted region patches with terrain appearances
    region_patches = []

    # Create patches for each region type with realistic terrain appearance
    for i, region in enumerate(hd.weighted_regions):
        # Create polygon patch with terrain appearance
        patch = patches.Polygon(region.vertices, color=region.color, 
                              alpha=0.7, fill=True, zorder=2,
                              edgecolor='black', linewidth=0.5)
        region_patches.append(patch)
        ax.add_patch(patch)

        # Add terrain-specific patterns
        vertices = np.array(region.vertices)
        x_min, y_min = np.min(vertices, axis=0)
        x_max, y_max = np.max(vertices, axis=0)

        # Add terrain-specific patterns
        if region.color == 'darkgreen':  # Forest
            # Add tree-like symbols
            tree_count = 20
            np.random.seed(i)  # For consistent random patterns
            for _ in range(tree_count):
                tree_x = np.random.uniform(x_min+0.5, x_max-0.5)
                tree_y = np.random.uniform(y_min+0.5, y_max-0.5)
                # Tree trunk
                ax.plot([tree_x, tree_x], [tree_y-0.2, tree_y+0.1], 'saddlebrown', linewidth=1.5, solid_capstyle='round')
                # Tree foliage
                ax.scatter(tree_x, tree_y+0.3, s=50, color='forestgreen', marker='^', zorder=3)
            terrain_type = 'Dense Forest'
            speed_text = '0.5x'

        elif region.color == 'dimgray':  # Rocky terrain
            # Add rock-like symbols
            rock_count = 25
            np.random.seed(i+10)  # For consistent random patterns
            for _ in range(rock_count):
                rock_x = np.random.uniform(x_min+0.3, x_max-0.3)
                rock_y = np.random.uniform(y_min+0.3, y_max-0.3)
                rock_size = np.random.uniform(5, 15)
                ax.scatter(rock_x, rock_y, s=rock_size, color='gray', marker='o', zorder=3, alpha=0.8)
            terrain_type = 'Rocky Terrain'
            speed_text = '0.8x'

        elif region.color == 'olive':  # Marsh/Swamp
            # Add marsh-like patterns
            for w in range(5):
                y_pos = y_min + (y_max - y_min) * (w + 0.5) / 5
                x_vals = np.linspace(x_min, x_max, 100)
                y_vals = y_pos + 0.1 * np.sin(3 * (x_vals - x_min) / (x_max - x_min) * 2 * np.pi)
                ax.plot(x_vals, y_vals, color='darkkhaki', alpha=0.6, linestyle=':', linewidth=0.8, zorder=3)

            # Add some vegetation dots
            veg_count = 30
            np.random.seed(i+20)  # For consistent random patterns
            for _ in range(veg_count):
                veg_x = np.random.uniform(x_min+0.2, x_max-0.2)
                veg_y = np.random.uniform(y_min+0.2, y_max-0.2)
                ax.scatter(veg_x, veg_y, s=3, color='darkkhaki', marker='*', zorder=3)
            terrain_type = 'Marsh/Swamp'
            speed_text = '1.2x'

        else:  # Moon surface/road (lightgray)
            # Add crater-like patterns for moon surface
            crater_count = 15
            np.random.seed(i+30)  # For consistent random patterns
            for _ in range(crater_count):
                crater_x = np.random.uniform(x_min+0.5, x_max-0.5)
                crater_y = np.random.uniform(y_min+0.5, y_max-0.5)
                crater_size = np.random.uniform(0.2, 0.5)
                circle = plt.Circle((crater_x, crater_y), crater_size, fill=False, 
                                    edgecolor='darkgray', linestyle='-', linewidth=0.8, alpha=0.7)
                ax.add_patch(circle)
            terrain_type = 'Moon Surface'
            speed_text = '1.8x'

        # Calculate center of region for text placement
        center_x = sum(v[0] for v in region.vertices) / len(region.vertices)
        center_y = sum(v[1] for v in region.vertices) / len(region.vertices)

        # Add terrain type label
        ax.text(center_x, center_y, f'{terrain_type}\nSpeed: {speed_text}', 
                horizontalalignment='center',
                verticalalignment='center',
                color='white',
                fontweight='bold',
                fontsize=8,
                bbox=dict(facecolor='black', 
                         alpha=0.6, 
                         edgecolor='white',
                         boxstyle='round,pad=0.3',
                         linewidth=0.5))

    # Initialize obstacle patches with enhanced visibility
    obstacle_patches = []
    for obstacle in hd.obstacles:
        vertices = obstacle.get_vertices()
        patch = patches.Polygon(vertices, 
                              facecolor='darkgray',
                              edgecolor='black',
                              alpha=0.7,
                              linewidth=1.5,
                              zorder=3)
        obstacle_patches.append(patch)
        ax.add_patch(patch)

    ax.add_patch(robot_circle)
    ax.add_patch(goal_circle)

    def init():
        return [robot_circle, goal_circle, goal_marker, goal_marker2, path_line] + region_patches + obstacle_patches + lidar_lines

    def animate(frame):
        # Update simulation
        goal_reached = hd.step()

        # Update robot position and add velocity vector
        robot_circle.center = (hd.robot.x, hd.robot.y)

        # Update goal position (cross)
        goal_circle.center = (hd.goal[0], hd.goal[1])
        goal_marker.set_xdata([hd.goal[0]-cross_size, hd.goal[0]+cross_size])
        goal_marker.set_ydata([hd.goal[1], hd.goal[1]])
        goal_marker2.set_xdata([hd.goal[0], hd.goal[0]])
        goal_marker2.set_ydata([hd.goal[1]-cross_size, hd.goal[1]+cross_size])

        # Update path history with fade effect
        path_x = [p[0] for p in hd.robot.path_history]
        path_y = [p[1] for p in hd.robot.path_history]
        path_line.set_data(path_x, path_y)

        # Update obstacle positions with motion trails
        for obstacle, patch in zip(hd.obstacles, obstacle_patches):
            patch.set_xy(obstacle.vertices)
            # Add velocity vector to obstacles
            if obstacle.velocity != (0, 0):
                center = np.mean(obstacle.vertices, axis=0)
                vel_mag = np.linalg.norm(obstacle.velocity)
                if vel_mag > 0:
                    vel_dir = np.array(obstacle.velocity) / vel_mag
                    ax.arrow(center[0], center[1], 
                            vel_dir[0], vel_dir[1],
                            head_width=0.2, head_length=0.3,
                            fc='red', ec='red', alpha=0.5)

        # Update LIDAR visualization with enhanced visibility
        lidar_data = hd.robot.scan_environment(hd.obstacles, hd.weighted_regions)
        for i, (line, angle) in enumerate(zip(lidar_lines, hd.robot.lidar_angles)):
            # Calculate LIDAR end points
            end_x = hd.robot.x + lidar_data[i] * math.cos(angle + hd.robot.orientation)
            end_y = hd.robot.y + lidar_data[i] * math.sin(angle + hd.robot.orientation)

            # Set line data with gradient alpha
            # Start more transparent near robot, more visible at detection points
            line.set_data([hd.robot.x, end_x], [hd.robot.y, end_y])

            # Make LIDAR lines more visible when they detect obstacles
            if lidar_data[i] < hd.robot.lidar_range:
                line.set_alpha(0.6)  # More visible when detecting something
                line.set_color('orange')  # Change color for detection
            else:
                line.set_alpha(0.2)  # More transparent for no detection
                line.set_color('yellow')

        # Add status text
        status_text = f'Time: {frame*0.05:.1f}s\n'
        status_text += f'Robot Pos: ({hd.robot.x:.1f}, {hd.robot.y:.1f})\n'
        status_text += f'Goal Pos: ({hd.goal[0]:.1f}, {hd.goal[1]:.1f})\n'
        status_text += f'Distance: {np.linalg.norm([hd.robot.x - hd.goal[0], hd.robot.y - hd.goal[1]]):.2f}m'
        if goal_reached:
            status_text += '\nGOAL REACHED!'

        # Clear previous text
        if hasattr(animate, 'status_artist'):
            animate.status_artist.remove()
        # Add new text
        animate.status_artist = ax.text(0.02, 0.98, status_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        return [robot_circle, goal_circle, goal_marker, goal_marker2, path_line] + obstacle_patches + lidar_lines
    
    # Create frames directory if it doesn't exist
    import os
    import shutil
    frames_dir = 'terrain_simulation_frames'
    if os.path.exists(frames_dir):
        # Remove existing frames to ensure we generate new ones
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    
    # Function to save a frame
    def save_frame(frame_number):
        # Update the animation
        animate(frame_number)
        
        # Save the frame
        frame_file = os.path.join(frames_dir, f'frame_{frame_number:04d}.png')
        plt.savefig(frame_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        
        # Print progress
        if frame_number % 50 == 0:
            print(f'Saved frame {frame_number}')
    
    # Add detailed legend with units
    legend_elements = [
        plt.Line2D([0], [0], color='black', marker='o', label='Robot (Point)',
                   markersize=5, linewidth=0),
        plt.Line2D([0], [0], color='red', marker='+', label='Goal (Cross)',
                   markersize=8, linewidth=2),
        plt.Line2D([0], [0], color='blue', label='Robot Path', alpha=0.7),
        patches.Patch(facecolor='darkgreen', alpha=0.7, label='Dense Forest (0.5x speed)'),
        patches.Patch(facecolor='dimgray', alpha=0.7, label='Rocky Terrain (0.8x speed)'),
        patches.Patch(facecolor='olive', alpha=0.7, label='Marsh/Swamp (1.2x speed)'),
        patches.Patch(facecolor='lightgray', alpha=0.7, label='Moon Surface (1.8x speed)'),
        patches.Patch(facecolor='gray', alpha=0.5, label='Obstacles'),
        plt.Line2D([0], [0], color='y', label='LIDAR Rays', alpha=0.4)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add detailed title and subtitle with Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.suptitle('HD-BUG Algorithm Simulation with Realistic Environment', fontsize=14, y=0.95)
    plt.title('Path Planning with Forest, Rock, Marsh and Moon Surface Terrain', 
              fontsize=10, pad=10)
    
    # Adjust layout to prevent text overlap
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    print('Saving frames... This may take a few minutes.')
    
    # Save frames
    total_frames = 600  # Significantly increased number of frames to ensure goal is reached
    for frame in range(total_frames):
        # Check if goal is reached and stop early if it is
        if frame > 50 and hd.reached_goal:  # Allow some extra frames after reaching goal
            print(f"Goal reached at frame {frame-50}, stopping animation")
            break
        save_frame(frame)
    
    print(f'\nAll frames saved in {frames_dir}/')
    print(f'Total frames: {total_frames}')
    
    # Create animation from saved frames with a new filename
    print('\nCreating video from frames...')
    video_filename = 'hd_bug_terrain_simulation.mp4'
    os.system(f'ffmpeg -y -framerate 30 -i {frames_dir}/frame_%04d.png '
             f'-vf "scale=1514:1714" -c:v libx264 -pix_fmt yuv420p -crf 23 {video_filename}')
    print(f'\nVideo saved as {video_filename}')
    
    # Display final frame
    plt.show()

if __name__ == '__main__':
    main()