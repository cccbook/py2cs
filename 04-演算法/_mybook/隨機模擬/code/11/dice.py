import numpy as np

def roll_dice():
    return np.random.randint(1, 7) + np.random.randint(1, 7)

def simulate_game(num_games):
    rewards = []
    
    for _ in range(num_games):
        score = roll_dice()
        if score == 7:
            rewards.append(10)
        elif score == 2 or score == 12:
            rewards.append(5)
        else:
            rewards.append(0)
    
    return rewards

# 設定遊戲次數
num_games = 10000
game_rewards = simulate_game(num_games)

print(f"Average reward over {num_games} games: {np.mean(game_rewards)}")
print(f"Total reward: {np.sum(game_rewards)}")
