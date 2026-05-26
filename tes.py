import numpy as np

# Jumlah kotak
n_states = 12

# Matriks reward (R)
R = np.full((n_states, n_states), -1)  # default -1 (penalti langkah)

# Definisikan koneksi antar kotak (misal: linear, bisa diubah sesuai kebutuhan)
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)
]

# Set reward untuk koneksi
for (i, j) in connections:
    R[i, j] = 0  # langkah biasa
    R[j, i] = 0  # dua arah

# Set reward untuk goal
R[10, 11] = 100
R[11, 11] = 100  # goal ke goal

# Hyperparameters
gamma = 0.8  # discount factor
alpha = 0.9  # learning rate
episodes = 1000
epochs = 10

def get_optimal_path(Q, start, goal):
    path = [start]
    state = start
    while state != goal:
        next_state = np.argmax(Q[state, :])
        path.append(next_state)
        state = next_state
        if len(path) > 20:  # Hindari loop tak berujung
            break
    return path

for epoch in range(epochs):
    # Inisialisasi ulang Q-table setiap epoch
    Q = np.zeros((n_states, n_states))
    for episode in range(episodes):
        state = np.random.randint(0, n_states)
        while state != 11:  # goal di 11
            possible_actions = np.where(R[state, :] >= 0)[0]
            action = np.random.choice(possible_actions)
            next_state = action
            reward = R[state, action]
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )
            state = next_state
    optimal_path = get_optimal_path(Q, 0, 11)
    print(f"Epoch {epoch+1}: Optimal path dari A ke L:", [chr(65 + x) for x in optimal_path])