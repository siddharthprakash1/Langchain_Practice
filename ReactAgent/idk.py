import cv2
import numpy as np
import random
import math
from numba import jit, prange

X, Y = 1920, 1080
num_particles = 10000
num_types = 5
dt = 0.009

window = np.zeros((Y, X, 3), dtype=np.uint8)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

colour_dict = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Yellow
    4: (0, 255, 255)   # Cyan
}

particle_types = np.random.randint(0, num_types, num_particles)
particle_pos_x = np.random.randint(5, X-5, num_particles).astype(np.float32)
particle_pos_y = np.random.randint(5, Y-5, num_particles).astype(np.float32)
p_v_x = np.random.randn(num_particles) * 0.1
p_v_y = np.random.randn(num_particles) * 0.1
masses = np.random.uniform(0.5, 1.5, num_particles)

forces = np.array([
    [ 1.5,  1.2,  1.2,  1.2,  1.2],
    [ 1.2,  1.5,  1.2,  1.2,  1.2],
    [ 1.2,  1.2,  1.5,  1.2,  1.2],
    [ 1.2,  1.2,  1.2,  1.5,  1.2],
    [ 1.2,  1.2,  1.2,  1.2,  1.5]
])

r = 200  # Increased interaction radius
repulsion_distance = 8
repulsion_strength = 1.0
friction_factor = 0.95

@jit(nopython=True, parallel=True)
def update_particles(particle_pos_x, particle_pos_y, p_v_x, p_v_y, particle_types, masses, forces, dt, friction_factor, X, Y):
    num_particles = len(particle_pos_x)
    for i in prange(num_particles):
        tot_force_x, tot_force_y = 0.0, 0.0
        for j in range(num_particles):
            if i != j:
                dx = particle_pos_x[j] - particle_pos_x[i]
                dy = particle_pos_y[j] - particle_pos_y[i]
                dist = math.sqrt(dx*dx + dy*dy)
                if 0 < dist < r:
                    if dist < repulsion_distance:
                        repulsion = repulsion_strength * (1 - dist/repulsion_distance)
                        tot_force_x -= dx / dist * repulsion
                        tot_force_y -= dy / dist * repulsion
                    else:
                        force = forces[particle_types[i], particle_types[j]] * (1 - (dist-repulsion_distance)/(r-repulsion_distance)) * (masses[j] / masses[i])
                        tot_force_x += dx / dist * force
                        tot_force_y += dy / dist * force
        p_v_x[i] = p_v_x[i] * friction_factor + tot_force_x * dt
        p_v_y[i] = p_v_y[i] * friction_factor + tot_force_y * dt
        particle_pos_x[i] += p_v_x[i]
        particle_pos_y[i] += p_v_y[i]
        # Wrap positions within bounds
        if particle_pos_x[i] < 0:
            particle_pos_x[i] += X
        if particle_pos_x[i] >= X:
            particle_pos_x[i] -= X
        if particle_pos_y[i] < 0:
            particle_pos_y[i] += Y
        if particle_pos_y[i] >= Y:
            particle_pos_y[i] -= Y

trail = np.zeros((Y, X), dtype=np.float32)

while True:
    window.fill(0)
    trail *= 0.95
    
    update_particles(particle_pos_x, particle_pos_y, p_v_x, p_v_y, particle_types, masses, forces, dt, friction_factor, X, Y)
    
    for i in range(num_particles):
        x, y = int(particle_pos_x[i]), int(particle_pos_y[i])
        if x >= 0 and x < X and y >= 0 and y < Y:
            trail[y, x] = 1
        color = colour_dict[particle_types[i]]
        cv2.circle(window, (x, y), 3, color, -1)
    
    trail_display = (trail * 255).astype(np.uint8)
    trail_color = cv2.applyColorMap(trail_display, cv2.COLORMAP_HOT)
    window = cv2.addWeighted(window, 1, trail_color, 0.3, 0)
    
    cv2.imshow("window", window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()