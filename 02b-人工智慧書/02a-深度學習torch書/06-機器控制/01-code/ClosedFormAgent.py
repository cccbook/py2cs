class ClosedFormAgent:
    def __init__(self, _):
        pass

    def reset(self, mode=None):
        pass

    def step(self, observation, reward, terminated):
        position, velocity, angle, angle_velocity = observation
        action = int(3. * angle + angle_velocity > 0.)
        return action

    def close(self):
        pass

