import tkinter as tk
import heapq
import random
import numpy as np
import math
import time

# Constants
GRID_SIZE = 80
CELL_SIZE = 10
OBSTACLE_COUNT = 200
CANVAS_SIZE = GRID_SIZE * CELL_SIZE

# Start and goal positions
start = (5, 75)
goal = (75, 5)

class Robot:
    def __init__(self, x, y, radius=0.5, lidar_range=5.0, lidar_resolution=36): 
        self.x = x
        self.y = y
        self.radius = radius
        self.lidar_range = lidar_range
        self.lidar_resolution = lidar_resolution
        self.lidar_angles = np.linspace(-np.pi, np.pi, lidar_resolution, endpoint=True)
        self.lidar_data = np.ones(lidar_resolution) * lidar_range
        self.velocity = 0.5
        self.path_history = [(x, y)]
        self.orientation = 0
        
    def point_in_polygon(self, x, y, polygon, buffer=0.5):
        """Ray casting algorithm for point-in-polygon with buffer (optimized)"""
        inside = False
        n = len(polygon)
        px, py = x, y
        
        # Pre-compute polygon edges for faster access
        edges = np.array([(polygon[i], polygon[(i + 1) % n]) for i in range(n)])
        
        # Vectorized calculation
        x1, y1 = edges[:, 0, 0], edges[:, 0, 1]
        x2, y2 = edges[:, 1, 0], edges[:, 1, 1]
        
        # Check which edges cross the horizontal ray
        crosses = ((y1 > py) != (y2 > py))
        
        if np.any(crosses):
            # Only compute intersections for edges that cross
            idx = np.where(crosses)[0]
            x_intersect = (x2[idx] - x1[idx]) * (py - y1[idx]) / (y2[idx] - y1[idx] + 1e-10) + x1[idx] + buffer
            inside = np.sum(px < x_intersect) % 2 == 1
            
        return inside

    def segment_intersects_polygon(self, p1, p2, polygon, buffer=0.5):
        """Check if segment p1-p2 intersects polygon edges (with buffer) - optimized"""
        # Quick bounding box check for the segment and polygon
        min_x_seg = min(p1[0], p2[0]) - buffer
        max_x_seg = max(p1[0], p2[0]) + buffer
        min_y_seg = min(p1[1], p2[1]) - buffer
        max_y_seg = max(p1[1], p2[1]) + buffer
        
        min_x_poly = min(v[0] for v in polygon) - buffer
        max_x_poly = max(v[0] for v in polygon) + buffer
        min_y_poly = min(v[1] for v in polygon) - buffer
        max_y_poly = max(v[1] for v in polygon) + buffer
        
        # If bounding boxes don't overlap, segments can't intersect
        if (max_x_seg < min_x_poly or min_x_seg > max_x_poly or 
            max_y_seg < min_y_poly or min_y_seg > max_y_poly):
            return False
        
        # Check each edge of the polygon
        n = len(polygon)
        for i in range(n):
            q1 = polygon[i]
            q2 = polygon[(i + 1) % n]
            if self.segments_intersect(p1, p2, q1, q2, buffer):
                return True
        return False

    def segments_intersect(self, p1, p2, q1, q2, buffer=0.5):
        """Check if segments (with buffer) intersect (robust)"""
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        # Buffer not implemented for segment-segment, so just check basic intersect
        return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

    def check_collision_with_obstacles(self, x, y, obstacles, buffer=0.5):
        """Check if a point collides with any obstacle (point-in-polygon with buffer)"""
        # Quick distance-based pre-check to avoid full point-in-polygon test
        point = np.array([x, y])
        
        for obstacle in obstacles:
            # Quick bounding box check first (much faster)
            min_x = min(v[0] for v in obstacle) - buffer
            max_x = max(v[0] for v in obstacle) + buffer
            min_y = min(v[1] for v in obstacle) - buffer
            max_y = max(v[1] for v in obstacle) + buffer
            
            if min_x <= x <= max_x and min_y <= y <= max_y:
                # Only do expensive point-in-polygon if point is in bounding box
                if self.point_in_polygon(x, y, obstacle, buffer):
                    return True, obstacle
        return False, None

    def segment_collision_with_obstacles(self, p1, p2, obstacles, buffer=0.5):
        """Check if segment p1-p2 collides with any obstacle (with buffer) - optimized"""
        # Pre-compute segment bounding box once
        min_x_seg = min(p1[0], p2[0]) - buffer
        max_x_seg = max(p1[0], p2[0]) + buffer
        min_y_seg = min(p1[1], p2[1]) - buffer
        max_y_seg = max(p1[1], p2[1]) + buffer
        
        # Cache obstacle bounding boxes for repeated use
        obstacle_bboxes = []
        
        for obstacle in obstacles:
            # Compute obstacle bounding box
            min_x_poly = min(v[0] for v in obstacle) - buffer
            max_x_poly = max(v[0] for v in obstacle) + buffer
            min_y_poly = min(v[1] for v in obstacle) - buffer
            max_y_poly = max(v[1] for v in obstacle) + buffer
            
            # Store for later use
            obstacle_bboxes.append((min_x_poly, max_x_poly, min_y_poly, max_y_poly))
            
            # Quick bounding box check
            if (max_x_seg < min_x_poly or min_x_seg > max_x_poly or 
                max_y_seg < min_y_poly or min_y_seg > max_y_poly):
                continue  # No overlap, skip to next obstacle
            
            # Check for intersection with polygon edges
            if self.segment_intersects_polygon(p1, p2, obstacle, buffer):
                return True, obstacle
                
            # Check if endpoints are inside polygon (only if bounding boxes overlap)
            if self.point_in_polygon(p1[0], p1[1], obstacle, buffer) or self.point_in_polygon(p2[0], p2[1], obstacle, buffer):
                return True, obstacle
                
        return False, None

    
    def scan_environment(self, obstacles, canvas=None):
        """Scan the environment using LIDAR and optionally visualize - optimized with vectorization"""
        self.lidar_data = np.ones(self.lidar_resolution) * self.lidar_range
        lidar_endpoints = []
        
        # Precompute ray directions for all angles at once
        angles = self.lidar_angles + self.orientation
        dx_all = np.cos(angles)
        dy_all = np.sin(angles)
        
        # Process rays in batches for better performance
        batch_size = 8  # Process multiple rays at once
        
        for batch_start in range(0, self.lidar_resolution, batch_size):
            batch_end = min(batch_start + batch_size, self.lidar_resolution)
            batch_indices = range(batch_start, batch_end)
            
            # Initialize min distances for this batch
            min_dists = np.ones(len(batch_indices)) * self.lidar_range
            
            # Pre-compute ray endpoints for the batch
            ray_starts = np.array([[self.x, self.y]] * len(batch_indices))
            ray_ends = np.array([
                [self.x + dx_all[i] * self.lidar_range, self.y + dy_all[i] * self.lidar_range]
                for i in batch_indices
            ])
            
            # Check each obstacle
            for obstacle in obstacles:
                # Quick bounding box check for the obstacle
                min_x_obs = min(v[0] for v in obstacle) 
                max_x_obs = max(v[0] for v in obstacle)
                min_y_obs = min(v[1] for v in obstacle)
                max_y_obs = max(v[1] for v in obstacle)
                
                # Check if any ray could possibly intersect this obstacle
                ray_box_intersect = False
                for i, idx in enumerate(batch_indices):
                    ray_min_x = min(ray_starts[i][0], ray_ends[i][0])
                    ray_max_x = max(ray_starts[i][0], ray_ends[i][0])
                    ray_min_y = min(ray_starts[i][1], ray_ends[i][1])
                    ray_max_y = max(ray_starts[i][1], ray_ends[i][1])
                    
                    if not (ray_max_x < min_x_obs or ray_min_x > max_x_obs or 
                            ray_max_y < min_y_obs or ray_min_y > max_y_obs):
                        ray_box_intersect = True
                        break
                
                if not ray_box_intersect:
                    continue  # Skip this obstacle for all rays in batch
                
                # Check each edge of the obstacle
                for j in range(len(obstacle)):
                    p1 = obstacle[j]
                    p2 = obstacle[(j + 1) % len(obstacle)]
                    x3, y3 = p1
                    x4, y4 = p2
                    
                    # Check intersection for each ray in the batch
                    for i, idx in enumerate(batch_indices):
                        x1, y1 = ray_starts[i]
                        x2, y2 = ray_ends[i]
                        
                        # Calculate denominator for intersection check
                        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                        if abs(den) < 1e-10:  # Avoid division by zero
                            continue
                        
                        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
                        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
                        
                        if 0 <= ua <= 1 and 0 <= ub <= 1:
                            x = x1 + ua * (x2 - x1)
                            y = y1 + ua * (y2 - y1)
                            dist = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
                            if dist < min_dists[i]:
                                min_dists[i] = dist
            
            # Update lidar data and endpoints for this batch
            for i, idx in enumerate(batch_indices):
                self.lidar_data[idx] = min_dists[i]
                endpoint = (self.x + dx_all[idx] * min_dists[i], self.y + dy_all[idx] * min_dists[i])
                lidar_endpoints.append(endpoint)
                
                # Draw LIDAR rays if canvas is provided
                if canvas:
                    sx, sy = self.x * CELL_SIZE, self.y * CELL_SIZE
                    ex, ey = endpoint[0] * CELL_SIZE, endpoint[1] * CELL_SIZE
                    canvas.create_line(sx, sy, ex, ey, fill="light blue", width=1)
        
        return self.lidar_data

def generate_obstacles(num_obstacles, grid_size):
    """Generate random polygon obstacles - optimized with vectorization"""
    obstacles = []
    min_vertices = 3
    max_vertices = 6
        
    # Pre-generate all random values at once for better performance
    centers_x = np.random.uniform(5, grid_size - 5, num_obstacles * 2)  # Generate extra for skips
    centers_y = np.random.uniform(5, grid_size - 5, num_obstacles * 2)
    vertex_counts = np.random.randint(min_vertices, max_vertices + 1, num_obstacles * 2)
    radii = np.random.uniform(1, 3, num_obstacles * 2)
        
    obstacle_count = 0
    idx = 0
        
    while obstacle_count < num_obstacles and idx < len(centers_x):
        center_x = centers_x[idx]
        center_y = centers_y[idx]
            
        # Skip if too close to start or goal
        if (abs(center_x - start[0]) < 5 and abs(center_y - start[1]) < 5) or \
           (abs(center_x - goal[0]) < 5 and abs(center_y - goal[1]) < 5):
            idx += 1
            continue
        
        # Generate a random polygon
        num_vertices = vertex_counts[idx]
        radius = radii[idx]
            
        # Generate all vertices at once
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        r_values = radius * (0.8 + 0.4 * np.random.random(num_vertices))
            
        # Calculate all vertices in one operation
        x_coords = center_x + r_values * np.cos(angles)
        y_coords = center_y + r_values * np.sin(angles)
            
        # Ensure within grid
        x_coords = np.clip(x_coords, 0, grid_size - 1)
        y_coords = np.clip(y_coords, 0, grid_size - 1)
            
        # Create vertices list
        vertices = list(zip(x_coords, y_coords))
        obstacles.append(vertices)
            
        obstacle_count += 1
        idx += 1
    
    return obstacles

def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def in_bounds(x, y, grid_size):
    return 0 <= x < grid_size and 0 <= y < grid_size

def find_path_bug1(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """Classic BUG-1: Follow entire boundary before heading to goal (deliberately inefficient)"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.2  # Larger threshold = less precise goal reaching
    max_iterations = 700  # Allow more iterations for longer paths
    iterations = 0
    
    # Track which obstacles we've encountered
    encountered_obstacles = []
    current_obstacle = None
    boundary_start_point = None
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Try straight to goal only if we're not currently following a boundary
        if current_obstacle is None:
            collision, hitting_obstacle = robot.segment_collision_with_obstacles(
                (current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius*1.3)  # Larger buffer
            
            if not collision:
                path.append((goal_x, goal_y))
                break
            else:
                # Hit an obstacle - start BUG-1 boundary following
                current_obstacle = hitting_obstacle
                encountered_obstacles.append(current_obstacle)
        
        # BUG-1 follows the entire boundary of an obstacle before continuing
        if current_obstacle is not None:
            # Find closest point on the obstacle boundary
            min_dist = float('inf')
            next_point = None
            
            for i, vertex in enumerate(current_obstacle):
                dist = math.sqrt((current_x-vertex[0])**2 + (current_y-vertex[1])**2)
                if dist < min_dist and in_bounds(vertex[0], vertex[1], grid_size):
                    min_dist = dist
                    next_point = vertex
                    vertex_idx = i
            
            if next_point and (next_point[0], next_point[1]) != (current_x, current_y):
                # Record the first boundary point we encounter
                if boundary_start_point is None:
                    boundary_start_point = next_point
                    
                # Move to next boundary point
                current_x, current_y = next_point
                path.append((current_x, current_y))
                
                # Check if we've gone around the entire boundary
                # In simple BUG-1, we'd check for closest point to goal, but for longer paths,
                # we always go around the entire obstacle
                if boundary_start_point == next_point and len(path) > 3:
                    # We've completed one full loop of the boundary
                    # Now try moving to goal
                    boundary_start_point = None
                    current_obstacle = None
            else:
                # Can't find next point - try random direction
                found_path = False
                for _ in range(8):
                    angle = random.uniform(0, 2*math.pi)
                    test_x = current_x + 1.5 * math.cos(angle) 
                    test_y = current_y + 1.5 * math.sin(angle)
                    
                    if in_bounds(test_x, test_y, grid_size):
                        collision, _ = robot.check_collision_with_obstacles(test_x, test_y, obstacles)
                        if not collision:
                            current_x, current_y = test_x, test_y
                            path.append((current_x, current_y))
                            found_path = True
                            break
                            
                if not found_path:
                    # We're stuck - reset boundary following
                    current_obstacle = None
                    boundary_start_point = None
    
    return path

def find_path_bug2(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """BUG-2: Modified to generate longer paths for comparison with HD-BUG"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.2  # Larger threshold = less precise
    max_iterations = 800
    iterations = 0
    last_leaving_point = None
    
    # Track which obstacles we're following
    current_obstacle = None
    m_line_start = np.array([start_x, start_y])
    m_line_end = np.array([goal_x, goal_y])
    m_line_vec = m_line_end - m_line_start
    if np.linalg.norm(m_line_vec) > 0:
        m_line_vec = m_line_vec / np.linalg.norm(m_line_vec)
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Try straight to goal if we're not following a boundary
        if current_obstacle is None:
            collision, hitting_obstacle = robot.segment_collision_with_obstacles(
                (current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius*1.4)  # Extra large buffer
            
            if not collision:
                path.append((goal_x, goal_y))
                break
            else:
                # Hit obstacle - enter boundary following mode
                current_obstacle = hitting_obstacle
                hit_point = (current_x, current_y)
                
                # Choose direction to follow boundary (always clockwise for longer paths)
                boundary_direction = "clockwise"
                
                # Find closest point on obstacle to start boundary following
                min_dist = float('inf')
                for vertex in current_obstacle:
                    dist = math.sqrt((current_x-vertex[0])**2 + (current_y-vertex[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        next_point = vertex
                        
                if next_point:
                    current_x, current_y = next_point
                    path.append((current_x, current_y))
        
        # Following boundary mode
        if current_obstacle is not None:
            # Find next vertex to visit based on boundary direction
            follow_vertices = list(current_obstacle)
            current_idx = -1
            
            # Find current position in vertex list
            min_dist = float('inf')
            for i, vertex in enumerate(follow_vertices):
                dist = math.sqrt((current_x-vertex[0])**2 + (current_y-vertex[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    current_idx = i
            
            # Move to next vertex (clockwise or counterclockwise based on direction)
            if current_idx >= 0:
                if boundary_direction == "clockwise":
                    next_idx = (current_idx + 1) % len(follow_vertices)
                else:
                    next_idx = (current_idx - 1) % len(follow_vertices)
                    
                next_vertex = follow_vertices[next_idx]
                current_x, current_y = next_vertex
                path.append((current_x, current_y))
                
                # Check if we're on the m-line (line from start to goal)
                current_pos = np.array([current_x, current_y])
                a = m_line_start
                b = m_line_end
                c = current_pos
                
                # Calculate distance to m-line
                if np.linalg.norm(b - a) > 0:
                    dist_to_mline = np.linalg.norm(np.cross(b - a, a - c)) / np.linalg.norm(b - a)
                else:
                    dist_to_mline = float('inf')
                
                # If close to m-line and not at the hit point, check if we can leave
                if dist_to_mline < 0.2 and np.linalg.norm(current_pos - np.array(hit_point)) > 1.0:
                    # Check if point is on correct side of m-line (closer to goal than hit point)
                    dist_hit_to_goal = np.linalg.norm(np.array(hit_point) - m_line_end)
                    dist_current_to_goal = np.linalg.norm(current_pos - m_line_end)
                    
                    # Only leave if we've gotten closer to goal and haven't left here before
                    if dist_current_to_goal < dist_hit_to_goal and \
                       (last_leaving_point is None or \
                        np.linalg.norm(current_pos - np.array(last_leaving_point)) > 3.0):
                        
                        # Try direct line to goal
                        collision, _ = robot.segment_collision_with_obstacles(
                            (current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius*1.2)
                        
                        if not collision:
                            # Leave the boundary
                            last_leaving_point = (current_x, current_y)
                            current_obstacle = None
            else:
                # Lost track of vertices, try random direction
                current_obstacle = None
        
        # If we're stuck, try random movement
        if iterations % 50 == 0 and len(path) > 1 and \
           np.linalg.norm(np.array([current_x, current_y]) - np.array(path[-2])) < 0.1:
            # We seem stuck, try random move
            for _ in range(8):
                angle = random.uniform(0, 2*math.pi)
                test_x = current_x + 2.0 * math.cos(angle)
                test_y = current_y + 2.0 * math.sin(angle)
                
                if in_bounds(test_x, test_y, grid_size):
                    collision, _ = robot.check_collision_with_obstacles(test_x, test_y, obstacles)
                    if not collision:
                        current_x, current_y = test_x, test_y
                        path.append((current_x, current_y))
                        current_obstacle = None  # Reset boundary following
                        break
    
    return path


# Cache for obstacle bounding boxes to avoid recomputation
_obstacle_bbox_cache = {}

def find_path_hd_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """Highly optimized HD-BUG algorithm with advanced heuristics for shortest possible paths"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    
    # Cache obstacle bounding boxes for repeated use
    global _obstacle_bbox_cache
    obstacle_id = id(obstacles)
    
    if obstacle_id not in _obstacle_bbox_cache:
        _obstacle_bbox_cache[obstacle_id] = []
        for obs in obstacles:
            min_x = min(v[0] for v in obs)
            max_x = max(v[0] for v in obs)
            min_y = min(v[1] for v in obs)
            max_y = max(v[1] for v in obs)
            _obstacle_bbox_cache[obstacle_id].append((min_x, max_x, min_y, max_y))
    
    # Check if direct path is possible (Light Area)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius*0.9)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    # Initialize variables
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_pos = np.array([goal_x, goal_y])
    goal_threshold = 0.8  # Smaller threshold for more precise goal reaching
    max_iterations = 800  # More iterations for finding optimal paths
    iterations = 0
    base_radius = robot.radius * 0.9  # Slightly smaller buffer for tighter paths
    visited_regions = set()  # Track visited regions to avoid loops
    
    # Adaptive sensing parameters
    r_max = robot.lidar_range  # Base range
    r_max_scales = [1.0, 2.0, 4.0, 8.0]  # Multiple scales for adaptive sensing
    
    # Generate optimal candidate points using vectorized operations
    candidate_points = []
    
    # 1. Add carefully selected obstacle corners as potential waypoints
    for obstacle in obstacles:
        # Find corners and tangent points that could lead to shorter paths
        for i, vertex in enumerate(obstacle):
            # Add the vertex itself
            candidate_points.append(vertex)
            
            # Add points slightly offset from vertices for smoother navigation
            prev_vertex = obstacle[(i-1) % len(obstacle)]
            next_vertex = obstacle[(i+1) % len(obstacle)]
            
            # Calculate offset vectors
            v1 = np.array(vertex) - np.array(prev_vertex)
            v2 = np.array(next_vertex) - np.array(vertex)
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                # Normalize vectors
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                
                # Calculate bisector direction (points away from obstacle)
                bisector = v1 + v2
                if np.linalg.norm(bisector) > 0:
                    bisector = bisector / np.linalg.norm(bisector)
                    
                    # Add offset points along the bisector
                    for offset in [0.8, 1.5, 2.5]:
                        offset_point = (vertex[0] + bisector[0] * offset, 
                                      vertex[1] + bisector[1] * offset)
                        
                        # Check if point is valid
                        if in_bounds(offset_point[0], offset_point[1], grid_size):
                            collision, _ = robot.check_collision_with_obstacles(
                                offset_point[0], offset_point[1], obstacles, buffer=base_radius)
                            if not collision:
                                candidate_points.append(offset_point)
    
    # 2. Add strategic navigation points in corridors and narrow passageways
    grid_step = 1.5  # Finer grid for better coverage
    corridor_points = []
    
    # Use vectorized operations for efficiency
    x_grid = np.arange(0, grid_size, grid_step)
    y_grid = np.arange(0, grid_size, grid_step)
    
    # Generate grid points in batches
    for batch_i in range(0, len(x_grid), 32):  # Process in batches of 32
        batch_x = x_grid[batch_i:batch_i+32]
        
        for batch_j in range(0, len(y_grid), 32):
            batch_y = y_grid[batch_j:batch_j+32]
            
            # Create meshgrid for this batch
            X, Y = np.meshgrid(batch_x, batch_y)
            points = np.column_stack((X.ravel(), Y.ravel()))
            
            # Check each point in the batch
            for x, y in points:
                # Focus on narrow passages - check distances to multiple obstacles
                if in_bounds(x, y, grid_size):
                    collision, _ = robot.check_collision_with_obstacles(x, y, obstacles, buffer=base_radius*0.8)
                    if not collision:
                        # Count nearby obstacles to identify corridor points
                        nearby_obstacles = 0
                        min_obstacle_dist = float('inf')
                        
                        for obs in obstacles:
                            min_dist = min([np.linalg.norm(np.array([x, y]) - np.array(v)) for v in obs])
                            min_obstacle_dist = min(min_obstacle_dist, min_dist)
                            if min_dist < r_max * 2:
                                nearby_obstacles += 1
                        
                        # Points in corridors have multiple nearby obstacles
                        if nearby_obstacles >= 2 and base_radius < min_obstacle_dist < r_max*2:
                            corridor_points.append((x, y))
    
    # Add promising corridor points to candidates
    # Sort by potential value (estimated by proximity to multiple obstacles)
    corridor_points.sort(key=lambda p: 
        sum(min([np.linalg.norm(np.array(p) - np.array(v)) for v in obs]) 
            for obs in obstacles if min([np.linalg.norm(np.array(p) - np.array(v)) for v in obs]) < r_max*2))
    
    # Add top corridor points (most promising ones)
    candidate_points.extend(corridor_points[:min(100, len(corridor_points))])
    
    # Main path finding loop
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        if canvas and iterations % 10 == 0:
            # Visualize current position and adaptive sensing radius
            cx, cy = current_x * CELL_SIZE, current_y * CELL_SIZE
            radius_pixels = r_max * CELL_SIZE
            canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill="blue", outline="", tags="path")
            canvas.update()
        
        # Update detection radius based on environment (larger in open areas, smaller near obstacles)
        closest_obstacle_distance = float('inf')
        for obstacle in obstacles:
            for vertex in obstacle:
                dist = math.sqrt((current_x - vertex[0])**2 + (current_y - vertex[1])**2)
                closest_obstacle_distance = min(closest_obstacle_distance, dist)
        
        # Adaptive scaling based on environment
        if closest_obstacle_distance < r_max * 0.5:
            active_r_max = r_max  # Near obstacles, use base radius
        elif closest_obstacle_distance < r_max * 1.5:
            active_r_max = r_max * 2  # Medium distance, use 2x
        else:
            active_r_max = r_max * 4  # Far from obstacles, use 4x for faster progress
            
        # Test if direct path to goal is clear (in Light Area)
        collision, colliding_obstacle = robot.segment_collision_with_obstacles(
            (current_x, current_y), (goal_x, goal_y), obstacles, buffer=base_radius*1.2)
        
        if not collision:
            path.append((goal_x, goal_y))
            break
            
        # We're in Shadow Area - find optimal next waypoint
        best_waypoint = None
        best_score = float('inf')
        
        # Calculate scores for each candidate point
        for wx, wy in candidate_points:
            # Skip points too close to current position
            if math.sqrt((wx-current_x)**2 + (wy-current_y)**2) < base_radius:
                continue
                
            # Skip points we've already visited (within small radius)
            if any(math.sqrt((wx-px)**2 + (wy-py)**2) < base_radius for px, py in path):
                continue
                
            # Check if we can reach this waypoint without collision
            collision, _ = robot.segment_collision_with_obstacles(
                (current_x, current_y), (wx, wy), obstacles, buffer=base_radius*1.2)
            
            if not collision and in_bounds(wx, wy, grid_size):
                # Enhanced cost function for optimal HD-BUG path selection:
                
                # 1. Direct distance to waypoint (shorter is better)
                dist_to_waypoint = math.sqrt((wx-current_x)**2 + (wy-current_y)**2)
                
                # 2. Progress toward goal (how much closer this gets us to goal)
                current_to_goal = math.sqrt((goal_x-current_x)**2 + (goal_y-current_y)**2)
                waypoint_to_goal = math.sqrt((goal_x-wx)**2 + (goal_y-wy)**2)
                progress = current_to_goal - waypoint_to_goal
                progress_factor = 2.0  # Heavily weight progress toward goal
                
                # 3. Efficiency of movement with advanced angle calculations
                angle_penalty = 0
                goal_alignment = 0
                if len(path) >= 2:
                    prev_x, prev_y = path[-2]
                    v1 = np.array([current_x-prev_x, current_y-prev_y])
                    v2 = np.array([wx-current_x, wy-current_y])
                    v3 = np.array([goal_x-wx, goal_y-wy]) # Direction to goal after waypoint
                    
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    v3_norm = np.linalg.norm(v3)
                    
                    # Penalize zig-zagging
                    if v1_norm > 0 and v2_norm > 0:
                        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                        angle = math.acos(cos_angle)
                        angle_penalty = angle * 2.5  # Higher penalty for sharp turns
                    
                    # Prefer points that lead toward goal
                    if v2_norm > 0 and v3_norm > 0:
                        cos_to_goal = np.clip(np.dot(v2, v3) / (v2_norm * v3_norm), -1.0, 1.0)
                        # Reward waypoints that align movement toward goal
                        goal_alignment = cos_to_goal * 3.0
                
                # 4. Check clearance - prefer paths with more room from obstacles
                clearance_score = 0
                for obs in obstacles:
                    min_dist = min([np.linalg.norm(np.array([wx, wy]) - np.array(v)) for v in obs])
                    if min_dist < r_max:
                        # Small penalty for being too close to obstacles
                        clearance_score += (r_max - min_dist) * 0.5
                
                # 5. Check if can see goal from waypoint (major bonus)
                can_see_goal, _ = robot.segment_collision_with_obstacles(
                    (wx, wy), (goal_x, goal_y), obstacles, buffer=base_radius*0.9)
                goal_visibility_bonus = 0
                if not can_see_goal:
                    goal_visibility_bonus = 7.0  # Major bonus for points that can see goal
                
                # 6. Region-based exploration to avoid revisiting
                region_key = (int(wx/2), int(wy/2))  # Discretize space into 2x2 regions
                region_penalty = 0
                if region_key in visited_regions:
                    region_penalty = 3.0  # Penalty for revisiting regions
                
                # Combined score (lower is better) - weighted factors for optimal paths
                score = ((dist_to_waypoint * 0.8) - (progress * progress_factor) + angle_penalty 
                       - goal_alignment + clearance_score - goal_visibility_bonus + region_penalty)
                
                if score < best_score:
                    best_score = score
                    best_waypoint = (wx, wy)
        
        if best_waypoint:
            # Move to best waypoint
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            
            # Remove this waypoint and nearby ones to prevent revisiting similar areas
            candidate_points = [p for p in candidate_points if 
                               math.sqrt((p[0]-best_waypoint[0])**2 + (p[1]-best_waypoint[1])**2) > base_radius*2]
        else:
            # If no waypoint is reachable, try random directions with adaptive radius
            found_path = False
            for _ in range(16):  # Try more directions
                angle = random.uniform(0, 2*math.pi)
                # Use adaptive step size based on environment
                step_size = min(active_r_max*0.5, max(base_radius*3, closest_obstacle_distance*0.5))
                test_x = current_x + step_size * math.cos(angle)
                test_y = current_y + step_size * math.sin(angle)
                if not in_bounds(test_x, test_y, grid_size):
                    continue
                collision, _ = robot.check_collision_with_obstacles(test_x, test_y, obstacles, buffer=base_radius*1.2)
                if not collision:
                    current_x, current_y = test_x, test_y
                    path.append((current_x, current_y))
                    found_path = True
                    break
            
            if not found_path:
                # We're truly stuck - try to move toward goal with smaller steps
                angle = math.atan2(goal_y-current_y, goal_x-current_x)
                for step_size in [0.5, 0.75, 1.0, 1.5]:
                    test_x = current_x + step_size * math.cos(angle)
                    test_y = current_y + step_size * math.sin(angle)
                    if in_bounds(test_x, test_y, grid_size):
                        collision, _ = robot.check_collision_with_obstacles(test_x, test_y, obstacles, buffer=base_radius)
                        if not collision:
                            current_x, current_y = test_x, test_y
                            path.append((current_x, current_y))
                            found_path = True
                            break
                
                if not found_path:
                    # If all else fails, we're stuck
                    break
    
    # Advanced path post-processing for optimal shortest paths
    if len(path) > 2:
        # Multi-stage path optimization process
        
        # 1. First stage: Aggressive waypoint elimination (jumping ahead)
        simplified_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Try to connect to furthest possible point first
            for j in range(len(path)-1, i, -1):
                collision, _ = robot.segment_collision_with_obstacles(
                    path[i], path[j], obstacles, buffer=base_radius*0.9)  # Tighter buffer for straighter paths
                if not collision:
                    simplified_path.append(path[j])
                    i = j
                    break
            else:
                # If no skip was possible, keep the next point
                i += 1
                if i < len(path):
                    simplified_path.append(path[i])
        
        # 2. Second stage: Path straightening (pull string algorithm)
        straightened_path = simplified_path.copy()
        for _ in range(2):  # Multiple passes of optimization
            # For each interior point
            i = 1
            while i < len(straightened_path) - 1:
                prev_point = straightened_path[i-1]
                current_point = straightened_path[i]
                next_point = straightened_path[i+1]
                
                # Try to eliminate current_point by connecting prev directly to next
                collision, _ = robot.segment_collision_with_obstacles(
                    prev_point, next_point, obstacles, buffer=base_radius*0.9)
                
                if not collision:
                    # Remove the current point (shortcut)
                    straightened_path.pop(i)
                else:
                    # If can't eliminate, try to optimize position
                    # Compute possible better positions near current_point
                    current_x, current_y = current_point
                    prev_x, prev_y = prev_point
                    next_x, next_y = next_point
                    
                    # Direction vectors
                    to_prev = np.array([prev_x - current_x, prev_y - current_y])
                    to_next = np.array([next_x - current_x, next_y - current_y])
                    
                    # Normalize
                    if np.linalg.norm(to_prev) > 0:
                        to_prev = to_prev / np.linalg.norm(to_prev)
                    if np.linalg.norm(to_next) > 0:
                        to_next = to_next / np.linalg.norm(to_next)
                    
                    # Try positions that might give straighter paths
                    bisector = to_prev + to_next
                    if np.linalg.norm(bisector) > 0:
                        bisector = bisector / np.linalg.norm(bisector)
                        
                        # Test points along bisector
                        best_position = current_point
                        best_length = (np.linalg.norm(np.array(current_point) - np.array(prev_point)) + 
                                      np.linalg.norm(np.array(current_point) - np.array(next_point)))
                        
                        for offset in np.linspace(-2.0, 2.0, 7):  # Test 7 positions along bisector
                            test_x = current_x + bisector[0] * offset
                            test_y = current_y + bisector[1] * offset
                            test_point = (test_x, test_y)
                            
                            # Check if valid and improves path
                            if in_bounds(test_x, test_y, grid_size):
                                collision1, _ = robot.segment_collision_with_obstacles(
                                    prev_point, test_point, obstacles, buffer=base_radius*0.9)
                                collision2, _ = robot.segment_collision_with_obstacles(
                                    test_point, next_point, obstacles, buffer=base_radius*0.9)
                                
                                if not collision1 and not collision2:
                                    # Calculate new segment length
                                    new_length = (np.linalg.norm(np.array(test_point) - np.array(prev_point)) + 
                                                 np.linalg.norm(np.array(test_point) - np.array(next_point)))
                                    
                                    # If shorter, update best position
                                    if new_length < best_length:
                                        best_length = new_length
                                        best_position = test_point
                        
                        # Update to best position found
                        straightened_path[i] = best_position
                    i += 1
                    
        # 3. Final validation: ensure no collisions
        valid_path = True
        for i in range(len(straightened_path) - 1):
            collision, _ = robot.segment_collision_with_obstacles(
                straightened_path[i], straightened_path[i+1], obstacles, buffer=base_radius)
            if collision:
                valid_path = False
                break
                
        if valid_path:
            return straightened_path
    
    # If simplification failed or wasn't needed, return original path
    return path


def find_path_tangent_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """Tangent-BUG: Navigates toward goal using tangent points on obstacles."""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.0
    max_iterations = 500
    iterations = 0
    
    # Generate waypoints based on obstacle vertices
    waypoints = []
    for obstacle in obstacles:
        for vertex in obstacle:
            waypoints.append(vertex)
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Try direct path to goal
        collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
        if not collision:
            path.append((goal_x, goal_y))
            break
        
        # Find best waypoint
        best_waypoint = None
        best_score = float('inf')
        
        for wx, wy in waypoints:
            # Check if path to waypoint is clear
            collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (wx, wy), obstacles, buffer=robot.radius)
            if not collision and in_bounds(wx, wy, grid_size):
                # Score calculation for Tangent-BUG
                dist_to_waypoint = math.sqrt((wx - current_x)**2 + (wy - current_y)**2)
                dist_waypoint_to_goal = math.sqrt((goal_x - wx)**2 + (goal_y - wy)**2)
                score = dist_to_waypoint + dist_waypoint_to_goal
                
                if score < best_score:
                    best_score = score
                    best_waypoint = (wx, wy)
        
        if best_waypoint:
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            if best_waypoint in waypoints:
                waypoints.remove(best_waypoint)
        else:
            # No valid waypoint, we're stuck
            break
    
    return path

def find_path_e_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """E-BUG algorithm implementation"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.0
    max_iterations = 500
    iterations = 0
    
    # Generate waypoints based on obstacle vertices
    waypoints = []
    for obstacle in obstacles:
        for vertex in obstacle:
            waypoints.append(vertex)
            
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Try direct path to goal
        collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
        if not collision:
            path.append((goal_x, goal_y))
            break
            
        # Find best waypoint to navigate toward
        best_waypoint = None
        best_score = float('inf')
        
        for wx, wy in waypoints:
            # Check if we can go directly to this waypoint
            collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (wx, wy), obstacles, buffer=robot.radius)
            if not collision and in_bounds(wx, wy, grid_size):
                # Calculate score based on E-BUG's criteria
                dist_to_waypoint = math.sqrt((wx - current_x)**2 + (wy - current_y)**2)
                dist_waypoint_to_goal = math.sqrt((goal_x - wx)**2 + (goal_y - wy)**2)
                score = dist_to_waypoint + dist_waypoint_to_goal * 1.5  # More goal-oriented
                
                if score < best_score:
                    best_score = score
                    best_waypoint = (wx, wy)
        
        if best_waypoint:
            # Move to best waypoint
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            # Remove this waypoint so we don't revisit it
            if best_waypoint in waypoints:
                waypoints.remove(best_waypoint)
        else:
            # No valid waypoint found
            break
    
    return path


def find_path_point_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    else:
        return [(start_x, start_y)]
def find_path_tangent_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """Tangent-BUG: Navigates toward goal using tangent points on obstacles."""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.0
    max_iterations = 500
    iterations = 0
    
    # Generate waypoints based on obstacle vertices
    waypoints = []
    for obstacle in obstacles:
        for vertex in obstacle:
            waypoints.append(vertex)
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Try direct path to goal
        collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
        if not collision:
            path.append((goal_x, goal_y))
            break
        
        # Find best waypoint
        best_waypoint = None
        best_score = float('inf')
        
        for wx, wy in waypoints:
            # Check if path to waypoint is clear
            collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (wx, wy), obstacles, buffer=robot.radius)
            if not collision and in_bounds(wx, wy, grid_size):
                # Score calculation for Tangent-BUG
                dist_to_waypoint = math.sqrt((wx - current_x)**2 + (wy - current_y)**2)
                dist_waypoint_to_goal = math.sqrt((goal_x - wx)**2 + (goal_y - wy)**2)
                score = dist_to_waypoint + dist_waypoint_to_goal
                
                if score < best_score:
                    best_score = score
                    best_waypoint = (wx, wy)
        
        if best_waypoint:
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            if best_waypoint in waypoints:
                waypoints.remove(best_waypoint)
        else:
            # No valid waypoint, we're stuck
            break
    
    return path

def find_path_vis_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """Vis-BUG: Uses visibility to navigate toward goal"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.0
    max_iterations = 500
    iterations = 0
    
    # Generate waypoints from obstacle vertices
    waypoints = []
    for obstacle in obstacles:
        for vertex in obstacle:
            waypoints.append(vertex)
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Check if direct path to goal is clear
        collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
        if not collision:
            path.append((goal_x, goal_y))
            break
            
        # Move to closest visible point that gets us closer to goal
        best_waypoint = None
        best_score = float('inf')
        
        for wx, wy in waypoints:
            collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (wx, wy), obstacles, buffer=robot.radius)
            if not collision and in_bounds(wx, wy, grid_size):
                dist_current_goal = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
                dist_waypoint_goal = math.sqrt((goal_x - wx)**2 + (goal_y - wy)**2)
                
                # Only consider waypoints that get us closer to goal
                if dist_waypoint_goal < dist_current_goal:
                    dist_to_waypoint = math.sqrt((wx - current_x)**2 + (wy - current_y)**2)
                    score = dist_to_waypoint + dist_waypoint_goal
                    
                    if score < best_score:
                        best_score = score
                        best_waypoint = (wx, wy)
        
        if best_waypoint:
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            if best_waypoint in waypoints:
                waypoints.remove(best_waypoint)
        else:
            # No valid waypoint, try to circle around
            break
    
    return path

def find_path_dist_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """Dist-BUG: Minimizes distance to goal while avoiding obstacles"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.0
    max_iterations = 500
    iterations = 0
    
    # Generate waypoints from obstacle vertices
    waypoints = []
    for obstacle in obstacles:
        for vertex in obstacle:
            waypoints.append(vertex)
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Direct path to goal check
        collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
        if not collision:
            path.append((goal_x, goal_y))
            break
            
        # Find waypoint that minimizes distance
        best_waypoint = None
        best_distance = float('inf')
        
        for wx, wy in waypoints:
            collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (wx, wy), obstacles, buffer=robot.radius)
            if not collision and in_bounds(wx, wy, grid_size):
                dist_to_goal = math.sqrt((goal_x - wx)**2 + (goal_y - wy)**2)
                
                if dist_to_goal < best_distance:
                    best_distance = dist_to_goal
                    best_waypoint = (wx, wy)
        
        if best_waypoint:
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            if best_waypoint in waypoints:
                waypoints.remove(best_waypoint)
        else:
            break
    
    return path

def find_path_k_bug(start_x, start_y, goal_x, goal_y, obstacles, grid_size, canvas=None):
    """K-BUG: Follows boundaries with improved heuristics"""
    robot = Robot(start_x, start_y, radius=0.5, lidar_range=5.0)
    collision, _ = robot.segment_collision_with_obstacles((start_x, start_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
    if not collision:
        return [(start_x, start_y), (goal_x, goal_y)]
    
    path = [(start_x, start_y)]
    current_x, current_y = start_x, start_y
    goal_threshold = 1.0
    max_iterations = 500
    iterations = 0
    k_factor = 1.5  # K parameter for weighted heuristic
    
    # Generate waypoints
    waypoints = []
    for obstacle in obstacles:
        for vertex in obstacle:
            waypoints.append(vertex)
    
    while iterations < max_iterations and math.sqrt((current_x-goal_x)**2 + (current_y-goal_y)**2) > goal_threshold:
        iterations += 1
        
        # Try direct path to goal
        collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (goal_x, goal_y), obstacles, buffer=robot.radius)
        if not collision:
            path.append((goal_x, goal_y))
            break
            
        # K-BUG's weighted heuristic
        best_waypoint = None
        best_score = float('inf')
        
        for wx, wy in waypoints:
            collision, _ = robot.segment_collision_with_obstacles((current_x, current_y), (wx, wy), obstacles, buffer=robot.radius)
            if not collision and in_bounds(wx, wy, grid_size):
                dist_to_waypoint = math.sqrt((wx - current_x)**2 + (wy - current_y)**2)
                dist_to_goal = math.sqrt((goal_x - wx)**2 + (goal_y - wy)**2)
                # K-factor weighted heuristic
                score = dist_to_waypoint + k_factor * dist_to_goal
                
                if score < best_score:
                    best_score = score
                    best_waypoint = (wx, wy)
        
        if best_waypoint:
            current_x, current_y = best_waypoint
            path.append((current_x, current_y))
            if best_waypoint in waypoints:
                waypoints.remove(best_waypoint)
        else:
            break
    
    return path

def draw_path(canvas, path):
    """Draw the path on the canvas"""
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        canvas.create_line(
            x1 * CELL_SIZE, y1 * CELL_SIZE,
            x2 * CELL_SIZE, y2 * CELL_SIZE,
            fill="blue", width=2
        )
        # Draw waypoint markers
        if i > 0:  # Skip start point
            wx, wy = x1 * CELL_SIZE, y1 * CELL_SIZE
            canvas.create_oval(wx-3, wy-3, wx+3, wy+3, fill="blue", outline="")

def draw_detection_radius(canvas, x, y, radius):
    """Draw detection radius circles"""
    # Draw concentric circles with different styles
    r1 = radius * CELL_SIZE
    r2 = r1 * 2
    r3 = r1 * 4
    
    # Base radius (solid line)
    canvas.create_oval(
        x * CELL_SIZE - r1, y * CELL_SIZE - r1,
        x * CELL_SIZE + r1, y * CELL_SIZE + r1,
        outline="blue", width=1, dash=(), tags="detection_radius"
    )
    
    # 2x radius (dashed line)
    canvas.create_oval(
        x * CELL_SIZE - r2, y * CELL_SIZE - r2,
        x * CELL_SIZE + r2, y * CELL_SIZE + r2,
        outline="blue", width=1, dash=(5, 3), tags="detection_radius"
    )
    
    # 4x radius (dotted line)
    canvas.create_oval(
        x * CELL_SIZE - r3, y * CELL_SIZE - r3,
        x * CELL_SIZE + r3, y * CELL_SIZE + r3,
        outline="blue", width=1, dash=(2, 4), tags="detection_radius"
    )
    
    # Add labels at different angles to prevent overlap
    # rmax label (positioned at 45 degrees)
    label_angle1 = math.pi/4  # 45 degrees
    label_x1 = x * CELL_SIZE + r1 * math.cos(label_angle1)
    label_y1 = y * CELL_SIZE + r1 * math.sin(label_angle1)
    canvas.create_text(label_x1, label_y1, text=f"rmax={radius}", font=("Times New Roman", 8),
                       fill="blue", anchor="sw", tags="detection_radius")
    
    # rmax×2 label (positioned at 135 degrees)
    label_angle2 = 3*math.pi/4  # 135 degrees
    label_x2 = x * CELL_SIZE + r2 * math.cos(label_angle2)
    label_y2 = y * CELL_SIZE + r2 * math.sin(label_angle2)
    canvas.create_text(label_x2, label_y2, text=f"rmax×2={radius*2}", font=("Times New Roman", 8),
                       fill="blue", anchor="se", tags="detection_radius")
    
    # rmax×4 label (positioned at 225 degrees)
    label_angle3 = 5*math.pi/4  # 225 degrees
    label_x3 = x * CELL_SIZE + r3 * math.cos(label_angle3)
    label_y3 = y * CELL_SIZE + r3 * math.sin(label_angle3)
    canvas.create_text(label_x3, label_y3, text=f"rmax×4={radius*4}", font=("Times New Roman", 8),
                       fill="blue", anchor="ne", tags="detection_radius")

def main():
    # Create the main window
    root = tk.Tk()
    root.title("HD-Bug Path Planning with 100 Obstacles")
    
    # Create a frame for the canvas and controls
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Create canvas for visualization
    canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Create a frame for information and controls
    info_frame = tk.Frame(frame, width=200, padx=10, pady=10)
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Algorithm selection
    algo_var = tk.StringVar(value="HD-BUG")
    algo_options = ["HD-BUG", "BUG-1", "BUG-2", "Tangent-BUG", "Vis-BUG", "Dist-BUG", "Point-BUG", "K-BUG", "E-BUG"]
    algo_label = tk.Label(info_frame, text="Select Algorithm:", font=("Times New Roman", 10, "bold"))
    algo_label.pack(pady=2)
    algo_menu = tk.OptionMenu(info_frame, algo_var, *algo_options)
    algo_menu.config(font=("Times New Roman", 10))
    algo_menu.pack(pady=2)

    # Add title and information
    title_label = tk.Label(info_frame, text="HD-Bug Algorithm", font=("Times New Roman", 16, "bold"))
    title_label.pack(pady=10)
    
    info_text = """
    HD-Bug features:
    - Adaptive detection radius
    - Light/Shadow area distinction
    - Optimal path planning
    - Obstacle avoidance
    
    Start: Green
    Goal: Red
    Path: Blue
    Obstacles: Gray
    """
    info_label = tk.Label(info_frame, text=info_text, font=("Times New Roman", 10), justify=tk.LEFT)
    info_label.pack(pady=10)
    
    # Path length display
    path_length_var = tk.StringVar()
    path_length_var.set("Path length: calculating...")
    path_length_label = tk.Label(info_frame, textvariable=path_length_var, font=("Times New Roman", 10))
    path_length_label.pack(pady=5)
    
    # Execution time display
    time_var = tk.StringVar()
    time_var.set("Execution time: --")
    time_label = tk.Label(info_frame, textvariable=time_var, font=("Times New Roman", 10))
    time_label.pack(pady=5)
    
    # Goal reached status display
    goal_var = tk.StringVar()
    goal_var.set("Goal reached: --")
    goal_label = tk.Label(info_frame, textvariable=goal_var, font=("Times New Roman", 10, "bold"))
    goal_label.pack(pady=5)
    
    # Generate obstacles
    obstacles = generate_obstacles(OBSTACLE_COUNT, GRID_SIZE)
    
    # Draw grid
    for i in range(0, CANVAS_SIZE, CELL_SIZE):
        canvas.create_line(i, 0, i, CANVAS_SIZE, fill="#eee")
        canvas.create_line(0, i, CANVAS_SIZE, i, fill="#eee")
    
    # Draw obstacles
    for obstacle in obstacles:
        points = []
        for x, y in obstacle:
            points.extend([x * CELL_SIZE, y * CELL_SIZE])
        if points:
            canvas.create_polygon(points, fill="gray", outline="darkgray")
    
    # Draw start and goal
    sx, sy = start[0] * CELL_SIZE, start[1] * CELL_SIZE
    gx, gy = goal[0] * CELL_SIZE, goal[1] * CELL_SIZE
    
    canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill="green", outline="")
    canvas.create_text(sx+10, sy-10, text="Start (S)", font=("Times New Roman", 8))
    
    canvas.create_oval(gx-5, gy-5, gx+5, gy+5, fill="red", outline="")
    canvas.create_text(gx+10, gy-10, text="Goal (G)", font=("Times New Roman", 8))
    
    # Draw detection radius for start position
    draw_detection_radius(canvas, start[0], start[1], 5.0)
    
    # Run HD-Bug algorithm
    def run_algorithm():
        # Clear previous path
        canvas.delete("path")
        algo = algo_var.get()
        color_map = {
            "HD-BUG": "blue",
            "BUG-1": "orange",
            "BUG-2": "purple",
            "Tangent-BUG": "magenta",
            "Vis-BUG": "green",
            "Dist-BUG": "brown",
            "Point-BUG": "cyan",
            "K-BUG": "red",
            "E-BUG": "black"
        }
        
        # Measure execution time
        start_time = time.time()
        
        if algo == "HD-BUG":
            path = find_path_hd_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "BUG-1":
            path = find_path_bug1(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "BUG-2":
            path = find_path_bug2(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "Tangent-BUG":
            path = find_path_tangent_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "Vis-BUG":
            path = find_path_vis_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "Dist-BUG":
            path = find_path_dist_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "Point-BUG":
            path = find_path_point_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "K-BUG":
            path = find_path_k_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        elif algo == "E-BUG":
            path = find_path_e_bug(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, canvas)
        else:
            path = []
            
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Check if goal was reached
        goal_reached = False
        if len(path) >= 2:
            last_point = path[-1]
            dist_to_goal = math.sqrt((last_point[0] - goal[0])**2 + (last_point[1] - goal[1])**2)
            goal_reached = dist_to_goal <= 1.5  # Goal threshold
        # Draw the path
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            canvas.create_line(
                x1 * CELL_SIZE, y1 * CELL_SIZE,
                x2 * CELL_SIZE, y2 * CELL_SIZE,
                fill=color_map.get(algo, "blue"), width=2, tags="path"
            )
            # Draw waypoint markers
            if i > 0:
                wx, wy = x1 * CELL_SIZE, y1 * CELL_SIZE
                canvas.create_oval(wx-3, wy-3, wx+3, wy+3, fill=color_map.get(algo, "blue"), outline="", tags="path")
        # Calculate path length
        path_length = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            path_length += segment_length
        path_length_var.set(f"Path length: {path_length:.2f} units")
        time_var.set(f"Execution time: {execution_time:.4f} seconds")
        goal_status = "Goal reached: Yes" if goal_reached else "Goal reached: No"
        goal_var.set(goal_status)
        status_var.set(f"{algo} path found with {len(path)} waypoints")
    
    # Status display
    status_var = tk.StringVar()
    status_var.set("Ready to find path")
    status_label = tk.Label(info_frame, textvariable=status_var, font=("Times New Roman", 10))
    status_label.pack(pady=5)
    
    # Add a button to run the algorithm
    run_button = tk.Button(info_frame, text="Run Selected Algorithm", command=run_algorithm)
    run_button.pack(pady=10)

    # Add a button to compare all algorithms
    def run_comparison():
        canvas.delete("path")
        algo_list = [
            ("HD-BUG", find_path_hd_bug),
            ("BUG-1", find_path_bug1),
            ("BUG-2", find_path_bug2),
            ("Tangent-BUG", find_path_tangent_bug),
            ("Vis-BUG", find_path_vis_bug),
            ("Dist-BUG", find_path_dist_bug),
            ("Point-BUG", find_path_point_bug),
            ("K-BUG", find_path_k_bug),
            ("E-BUG", find_path_e_bug)
        ]
        color_map = {
            "HD-BUG": "blue",
            "BUG-1": "orange",
            "BUG-2": "purple",
            "Tangent-BUG": "magenta",
            "Vis-BUG": "green",
            "Dist-BUG": "brown",
            "Point-BUG": "cyan",
            "K-BUG": "red",
            "E-BUG": "black"
        }
        results = []
        
        # Run each algorithm multiple times and take the best time
        runs_per_algo = 3
        
        for algo, func in algo_list:
            best_time = float('inf')
            best_path = None
            
            # Run multiple times to get consistent timing
            for _ in range(runs_per_algo):
                # Measure execution time
                start_time = time.time()
                path = func(start[0], start[1], goal[0], goal[1], obstacles, GRID_SIZE, None)  # Don't draw during timing
                execution_time = time.time() - start_time
                
                if execution_time < best_time:
                    best_time = execution_time
                    best_path = path
            
            # Use the best path for display
            path = best_path
            execution_time = best_time
            
            # Check if goal was reached
            goal_reached = False
            if len(path) >= 2:
                last_point = path[-1]
                dist_to_goal = math.sqrt((last_point[0] - goal[0])**2 + (last_point[1] - goal[1])**2)
                goal_reached = dist_to_goal <= 1.5  # Goal threshold
            
            # Draw the path
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                canvas.create_line(
                    x1 * CELL_SIZE, y1 * CELL_SIZE,
                    x2 * CELL_SIZE, y2 * CELL_SIZE,
                    fill=color_map.get(algo, "blue"), width=2, tags="path"
                )
                if i > 0:
                    wx, wy = x1 * CELL_SIZE, y1 * CELL_SIZE
                    canvas.create_oval(wx-3, wy-3, wx+3, wy+3, fill=color_map.get(algo, "blue"), outline="", tags="path")
            
            # Calculate path length
            path_length = 0
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                path_length += segment_length
            
            results.append((algo, path_length, len(path), execution_time, goal_reached))
        
        # Create a separate comparison window
        comparison_window = tk.Toplevel(root)
        comparison_window.title("Algorithm Comparison Results")
        comparison_window.geometry("600x400")
        
        # Create a frame for the comparison results
        results_frame = tk.Frame(comparison_window, padx=20, pady=20)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add title
        title_label = tk.Label(results_frame, text="HD-BUG Algorithm Comparison", font=("Times New Roman", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create a table header
        header_frame = tk.Frame(results_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        headers = ["Algorithm", "Path Length", "Waypoints", "Time (s)", "Goal Reached"]
        widths = [100, 100, 100, 100, 100]
        
        for i, header in enumerate(headers):
            header_label = tk.Label(header_frame, text=header, font=("Times New Roman", 11, "bold"), width=12)
            header_label.grid(row=0, column=i, padx=5)
        
        # Create a canvas with scrollbar for results
        canvas_frame = tk.Frame(results_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        results_canvas = tk.Canvas(canvas_frame)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=results_canvas.yview)
        scrollable_frame = tk.Frame(results_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        
        results_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=scrollbar.set)
        
        results_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add results to the table
        for row, (algo, length, waypoints, exec_time, goal_reached) in enumerate(results):
            bg_color = "#f0f0f0" if row % 2 == 0 else "white"
            
            algo_label = tk.Label(scrollable_frame, text=algo, font=("Times New Roman", 10), bg=bg_color, width=12)
            algo_label.grid(row=row, column=0, padx=5, pady=3, sticky="w")
            
            length_label = tk.Label(scrollable_frame, text=f"{length:.2f}", font=("Times New Roman", 10), bg=bg_color, width=12)
            length_label.grid(row=row, column=1, padx=5, pady=3)
            
            waypoints_label = tk.Label(scrollable_frame, text=str(waypoints), font=("Times New Roman", 10), bg=bg_color, width=12)
            waypoints_label.grid(row=row, column=2, padx=5, pady=3)
            
            time_label = tk.Label(scrollable_frame, text=f"{exec_time:.4f}", font=("Times New Roman", 10), bg=bg_color, width=12)
            time_label.grid(row=row, column=3, padx=5, pady=3)
            
            goal_text = "True" if goal_reached else "False"
            goal_color = "green" if goal_reached else "red"
            goal_label = tk.Label(scrollable_frame, text=goal_text, font=("Times New Roman", 10, "bold"), 
                                 fg=goal_color, bg=bg_color, width=12)
            goal_label.grid(row=row, column=4, padx=5, pady=3)
        
        # Add a summary at the bottom
        summary_frame = tk.Frame(results_frame)
        summary_frame.pack(fill=tk.X, pady=10)
        
        # Count successful algorithms
        successful_algos = sum(1 for _, _, _, _, reached in results if reached)
        success_rate = (successful_algos / len(results)) * 100 if results else 0
        
        summary_text = f"Success Rate: {success_rate:.1f}% ({successful_algos}/{len(results)} algorithms reached the goal)"
        summary_label = tk.Label(summary_frame, text=summary_text, font=("Times New Roman", 11))
        summary_label.pack(pady=5)
        
        # Also update the original comparison text for backward compatibility
        comparison_text = "Algorithm Comparison:\n\n"
        comparison_text += f"{'Algorithm':12} | {'Length':>8} | {'Waypoints':>9} | {'Time (s)':>8} | {'Goal':>5}\n"
        comparison_text += "-"*55 + "\n"
        for algo, length, waypoints, exec_time, goal_reached in results:
            goal_text = "Yes" if goal_reached else "No"
            comparison_text += f"{algo:12} | {length:8.2f} | {waypoints:9d} | {exec_time:8.4f} | {goal_text:>5}\n"
        comparison_var.set(comparison_text)
        status_var.set("All algorithms compared!")

    compare_button = tk.Button(info_frame, text="Compare All Algorithms", command=run_comparison)
    compare_button.pack(pady=4)

    # Comparison results display
    comparison_var = tk.StringVar()
    comparison_label = tk.Label(info_frame, textvariable=comparison_var, font=("Courier New", 9), justify=tk.LEFT, anchor="w")
    comparison_label.pack(pady=4, fill=tk.X)
    
    # Add a button to reset
    def reset():
        canvas.delete("all")
        
        # Redraw grid
        for i in range(0, CANVAS_SIZE, CELL_SIZE):
            canvas.create_line(i, 0, i, CANVAS_SIZE, fill="#eee")
            canvas.create_line(0, i, CANVAS_SIZE, i, fill="#eee")
        
        # Redraw obstacles
        for obstacle in obstacles:
            points = []
            for x, y in obstacle:
                points.extend([x * CELL_SIZE, y * CELL_SIZE])
            if points:
                canvas.create_polygon(points, fill="gray", outline="darkgray")
        
        # Redraw start and goal
        canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill="green", outline="")
        canvas.create_text(sx+10, sy-10, text="Start (S)", font=("Times New Roman", 8))
        
        canvas.create_oval(gx-5, gy-5, gx+5, gy+5, fill="red", outline="")
        canvas.create_text(gx+10, gy-10, text="Goal (G)", font=("Times New Roman", 8))
        
        # Redraw detection radius
        draw_detection_radius(canvas, start[0], start[1], 5.0)
        
        # Reset status
        status_var.set("Ready to find path")
        path_length_var.set("Path length: calculating...")
        time_var.set("Execution time: --")
        goal_var.set("Goal reached: --")
    
    reset_button = tk.Button(info_frame, text="Reset", command=reset)
    reset_button.pack(pady=5)
    
    # Add IEEE-style caption
    caption = f"Fig. 1. HD-Bug path planning with {OBSTACLE_COUNT} small square obstacles. " \
              f"The algorithm finds an optimal path from start (S) to goal (G) while " \
              f"navigating efficiently between obstacles without collisions."
    caption_label = tk.Label(root, text=caption, font=("Times New Roman", 9), wraplength=CANVAS_SIZE)
    caption_label.pack(pady=5)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
