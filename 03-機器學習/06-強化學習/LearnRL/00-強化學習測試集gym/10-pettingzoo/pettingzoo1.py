from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env()

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
    print(f'action={action}')
    env.step(action)
