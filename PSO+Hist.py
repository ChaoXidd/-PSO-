import numpy as np
import cv2
from skimage import exposure

class PSO:
    def __init__(self, n_particles, dim, bounds, max_iter, fitness_func):
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.fitness_func = fitness_func

        # 初始化粒子位置和速度
        self.particles = np.random.uniform(bounds[0][0], bounds[0][1], (n_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (n_particles, dim))
        self.personal_best = self.particles.copy()
        self.global_best = None

        self.personal_best_scores = np.full(n_particles, -np.inf)
        self.global_best_score = -np.inf

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.n_particles):
                fitness = self.fitness_func(self.particles[i])
                if fitness > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best[i] = self.particles[i]
                if fitness > self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = self.particles[i]

            # 更新速度和位置
            inertia = 0.5
            cognitive = 1.5
            social = 1.5
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (inertia * self.velocities[i] +
                                      cognitive * r1 * (self.personal_best[i] - self.particles[i]) +
                                      social * r2 * (self.global_best - self.particles[i]))
                self.particles[i] += self.velocities[i]

                # 边界检查
                self.particles[i] = np.clip(self.particles[i], self.bounds[0][0], self.bounds[0][1])
def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # 使用OpenCV自带的CLAHE（Contrast Limited Adaptive Histogram Equalization）算法
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

# 适应度函数
def fitness_function(params):
    # 提取参数
    clip_limit, grid_size_x, grid_size_y = params
    grid_size = (int(grid_size_x), int(grid_size_y))

    # 图像增强
    enhanced_image = adaptive_histogram_equalization(input_image, clip_limit, grid_size)

    # 计算图像的熵作为适应度（较高的熵表示较好的对比度和细节）
    hist = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist /= hist.sum()  # 归一化
    entropy = -np.sum(hist * np.log(hist + 1e-5))  # 避免log(0)

    return -entropy  # 熵越高，图像对比度越好

input_image = cv2.imread("jx_3.jpg", cv2.IMREAD_GRAYSCALE)
# bounds = [(1.0, 5.0), (4, 16), (4, 16)]  # 参数的边界: clip_limit, tile_grid_size_x, tile_grid_size_y
# pso = PSO(n_particles=30, dim=3, bounds=bounds, max_iter=50, fitness_func=fitness_function)
# pso.optimize()
# optimal_params = pso.global_best
# optimal_clip_limit = optimal_params[0]
# optimal_grid_size = (int(optimal_params[1]), int(optimal_params[2]))
# enhanced_image = adaptive_histogram_equalization(input_image, optimal_clip_limit, optimal_grid_size)
enhanced_image = adaptive_histogram_equalization(input_image, 2, (5,5))
#print(optimal_params)
cv2.imwrite("result2.png",enhanced_image)