import numpy as np


class RandomWalkMobility:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, position, velocity, dt, half_area, user_max_speed):
        speed = np.linalg.norm(velocity)
        if speed < 1e-12:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, user_max_speed)
            velocity = speed * np.array([np.cos(angle), np.sin(angle)])
        else:
            current_angle = np.arctan2(velocity[1], velocity[0])
            new_angle = current_angle + np.random.uniform(-np.pi / 6, np.pi / 6)
            speed_jitter = np.random.uniform(0.9, 1.1)
            speed = np.clip(speed * speed_jitter, 0.0, user_max_speed)
            velocity = speed * np.array([np.cos(new_angle), np.sin(new_angle)])
        new_xy = position + velocity * dt
        for i in range(2):
            if new_xy[i] < -half_area:
                new_xy[i] = -2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
            elif new_xy[i] > half_area:
                new_xy[i] = 2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
        return new_xy, velocity


class RandomWaypointMobility:
    def __init__(self, speed_min=0.5, speed_max=3.0, pause_steps=10):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pause_steps = pause_steps
        self.destination = None
        self.pause_counter = 0

    def reset(self):
        self.destination = None
        self.pause_counter = 0

    def _choose_destination(self, half_area):
        self.destination = np.random.uniform(-half_area, half_area, size=2)

    def step(self, position, velocity, dt, half_area, user_max_speed):
        if self.destination is None:
            if self.pause_counter > 0:
                self.pause_counter -= 1
                return position, np.zeros(2)
            self._choose_destination(half_area)
        direction = self.destination - position
        dist = np.linalg.norm(direction)
        if dist < 1.0:
            self.destination = None
            self.pause_counter = self.pause_steps
            return position, np.zeros(2)
        direction = direction / dist
        speed = np.random.uniform(self.speed_min, min(self.speed_max, user_max_speed))
        velocity = speed * direction
        new_xy = position + velocity * dt
        for i in range(2):
            if new_xy[i] < -half_area:
                new_xy[i] = -2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
                self.destination = None
            elif new_xy[i] > half_area:
                new_xy[i] = 2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
                self.destination = None
        return new_xy, velocity


class GaussMarkovMobility:
    def __init__(self, alpha=0.75, mean_speed=1.5, variance=1.0):
        self.alpha = alpha
        self.mean_speed = mean_speed
        self.variance = variance

    def reset(self):
        pass

    def step(self, position, velocity, dt, half_area, user_max_speed):
        speed = np.linalg.norm(velocity)
        if speed < 1e-12:
            angle = np.random.uniform(0, 2 * np.pi)
            speed_init = np.random.uniform(0.5, user_max_speed)
            velocity = speed_init * np.array([np.cos(angle), np.sin(angle)])
        current_speed = np.linalg.norm(velocity)
        if current_speed > 1e-12:
            direction = velocity / current_speed
        else:
            direction = np.array([1.0, 0.0])
        noise = np.random.normal(0.0, self.variance, size=2)
        mu = self.mean_speed * direction
        velocity = (self.alpha * velocity +
                    (1.0 - self.alpha) * mu +
                    np.sqrt(1.0 - self.alpha ** 2) * noise)
        final_speed = np.linalg.norm(velocity)
        if final_speed > user_max_speed:
            velocity = velocity / final_speed * user_max_speed
        new_xy = position + velocity * dt
        for i in range(2):
            if new_xy[i] < -half_area:
                new_xy[i] = -2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
            elif new_xy[i] > half_area:
                new_xy[i] = 2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
        return new_xy, velocity


class ConstantVelocityMobility:
    def __init__(self, speed=2.0, initial_angle=None):
        self.speed = speed
        self.initial_angle = initial_angle
        self._initialized = False

    def reset(self):
        self._initialized = False

    def step(self, position, velocity, dt, half_area, user_max_speed):
        if not self._initialized or np.linalg.norm(velocity) < 1e-12:
            angle = self.initial_angle if self.initial_angle is not None else np.random.uniform(0, 2 * np.pi)
            velocity = self.speed * np.array([np.cos(angle), np.sin(angle)])
            self._initialized = True
        new_xy = position + velocity * dt
        for i in range(2):
            if new_xy[i] < -half_area:
                new_xy[i] = -2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
            elif new_xy[i] > half_area:
                new_xy[i] = 2.0 * half_area - new_xy[i]
                velocity[i] *= -1.0
        return new_xy, velocity


MOBILITY_MODEL_MAP = {
    "random_walk": RandomWalkMobility,
    "random_waypoint": RandomWaypointMobility,
    "gauss_markov": GaussMarkovMobility,
    "constant_velocity": ConstantVelocityMobility,
}


def create_mobility_model(name: str, return_class=False, **kwargs):
    cls = MOBILITY_MODEL_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown mobility model: {name}. Options: {list(MOBILITY_MODEL_MAP.keys())}")
    if return_class:
        return cls, kwargs
    return cls(**kwargs)
