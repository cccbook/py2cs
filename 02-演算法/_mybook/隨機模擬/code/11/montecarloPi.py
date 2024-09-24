import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(num_samples):
    inside_circle = 0
    x_inside = []
    y_inside = []
    x_outside = []
    y_outside = []
    
    for _ in range(num_samples):
        x, y = np.random.uniform(-1, 1, 2)  # 隨機生成點
        distance = x**2 + y**2
        
        if distance <= 1:
            inside_circle += 1
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    pi_estimate = 4 * inside_circle / num_samples
    return pi_estimate, x_inside, y_inside, x_outside, y_outside

# 設定隨機樣本數
num_samples = 10000
pi_value, x_inside, y_inside, x_outside, y_outside = estimate_pi(num_samples)

print(f"Estimated value of pi: {pi_value}")

# 繪圖
plt.figure(figsize=(8, 8))
plt.scatter(x_inside, y_inside, color='blue', s=1, label='Inside Circle')
plt.scatter(x_outside, y_outside, color='red', s=1, label='Outside Circle')
plt.legend()
plt.title('Monte Carlo Simulation for Pi Estimation')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()
