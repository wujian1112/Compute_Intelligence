import random
import numpy as np
import matplotlib.pyplot as plt

# 定义函数 f
def f(x, y):
    return x**2 + y**2

# 遗传算法参数设置
population_size = 50  # 种群大小
chromosome_length = 2  # 染色体长度（二维坐标）
# 初始化种群
def create_population():
    population = []
    for _ in range(population_size):
        individual = [random.uniform(-10, 10)
                      for _ in range(chromosome_length)]
        population.append(individual)
    return population

# 评估种群中个体的适应度
def compute_fitness(population):
    fitness_scores = []
    for individual in population:
        x, y = individual[0], individual[1]
        fitness = 1/(f(x, y)+0.001)
        fitness_scores.append(fitness)
    return fitness_scores

# 选择操作 - 使用轮盘赌选择法
def selection(population, fitness_scores):
    selected_population = []
    total_fitness = sum(fitness_scores)
    while len(selected_population) < population_size:
        rand_num = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        for i, fitness in enumerate(fitness_scores):
            cumulative_fitness += fitness
            if cumulative_fitness > rand_num:
                selected_population.append(population[i])
                break
    return selected_population

# 交叉操作 - 使用单点交叉
def crossover(population):
    offspring = []
    for i in range(0, population_size, 2):
        parent1 = population[i]
        parent2 = population[i+1]
        crossover_point = random.randint(1, chromosome_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    return offspring

# 变异操作 - 在染色体上进行随机变异
def mutation(population):
    mutated_population = []
    for individual in population:
        if random.random() < 0.1:  # 变异概率为0.1
            mutated_individual = []
            for gene in individual:
                new_gene = gene + random.uniform(-0.1, 0.1)
                # 变异范围为[-0.1, 0.1]
                mutated_individual.append(new_gene)
            mutated_population.append(mutated_individual)
        else:
            mutated_population.append(individual)
    return mutated_population

# 主函数
def genetic_algorithm():
    population = create_population()
    generations = 100  # 迭代次数
    obj_value = []
    for _ in range(generations):
        temp_individual = max(population, key=lambda x: f(x[0], x[1]))
        obj_value.append(f(temp_individual[0], temp_individual[1]))
        fitness_scores = compute_fitness(population)
        selected_population = selection(population, fitness_scores)
        offspring = crossover(selected_population)
        mutated_population = mutation(offspring)
        population = mutated_population
    # 计算最好的个体
    best_individual = max(population, key=lambda x: f(x[0], x[1]))
    best_obj = f(best_individual[0], best_individual[1])
    print("最优解：", best_individual)
    print("最小值：", best_obj)
    plt.plot(obj_value)
    plt.show()

# 执行遗传算法
# 生成网格点数据
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
# 计算函数值
Z = f(X, Y)
# 绘制函数图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
# 设置图像标题
plt.title('f(X, Y) = X^2 +Y^2')
# 显示图像
plt.show()
genetic_algorithm()
