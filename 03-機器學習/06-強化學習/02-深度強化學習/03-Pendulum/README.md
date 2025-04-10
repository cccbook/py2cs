

* https://www.gymlibrary.dev/environments/classic_control/pendulum/


$$
r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)
$$

the minimum reward that can be obtained is -(pi2 + 0.1 * 82 + 0.001 * 22) = -16.2736044, while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).