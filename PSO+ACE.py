import numpy as np
import cv2
from skimage import exposure

# 初始化PSO参数
class PSO:
    def __init__(self, n_particles, dim, bounds, max_iter, fitness_func):
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.fitness_func = fitness_func

        # 初始化粒子位置和速度
        self.particles = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
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
                self.particles[i] = np.clip(self.particles[i], self.bounds[0], self.bounds[1])

# 图像对比度增强函数
def enhance_image(image, params):
    gamma, alpha = params
    brighter_image = alpha * np.power(image, gamma)  # 输出图像
    brighter_image = cv2.normalize(brighter_image,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    return brighter_image

# 适应度函数
def fitness_function(params):
    enhanced_image = enhance_image(input_image, params)
    hist = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist /= hist.sum()  # 归一化
    entropy = -np.sum(hist * np.log(hist + 1e-5))  # 避免log(0)
    return -entropy

input_image = cv2.imread("jx_3.jpg", cv2.IMREAD_GRAYSCALE)
bounds = [(0.01, 3), (0.5, 1)]  # 参数的边界
pso = PSO(n_particles=30, dim=2, bounds=bounds, max_iter=50, fitness_func=fitness_function)
pso.optimize()
optimal_params = pso.global_best
enhanced_image = enhance_image(input_image, optimal_params)
print(optimal_params)
cv2.imwrite("result1.png",enhanced_image)