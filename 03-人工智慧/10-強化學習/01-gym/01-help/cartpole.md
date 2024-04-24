# CartPole 

小車上有一個下端可擺動的車桿，請左右移動小車讓車桿不倒。

觀察環境會傳回四個浮點數值，請根據環境傳回值，決定要如何移動小車

```
 |  | Num | Observation           | Min                 | Max               |
 |  |-----|-----------------------|---------------------|-------------------|
 |  | 0   | Cart Position         | -4.8                | 4.8               |
 |  | 1   | Cart Velocity         | -Inf                | Inf               |
 |  | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
 |  | 3   | Pole Angular Velocity | -Inf                | Inf               |
```


```
$ python env_help.py CartPole-v1
Help on CartPoleEnv in module gymnasium.envs.classic_control.cartpole object:

class CartPoleEnv(gymnasium.core.Env)
 |  CartPoleEnv(render_mode: Optional[str] = None)
 |
 |  ## Description
 |
 |  This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
 |  ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
 |  A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
 |  The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
 |   in the left and right direction on the cart.
 |
 |  ## Action Space
 |
 |  The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
 |   of the fixed force the cart is pushed with.
 |
 |  - 0: Push cart to the left
 |  - 1: Push cart to the right
 |
 |  **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
 |   the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
 |
 |  ## Observation Space
 |
 |  The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
 |
 |  | Num | Observation           | Min                 | Max               |
 |  |-----|-----------------------|---------------------|-------------------|
 |  | 0   | Cart Position         | -4.8                | 4.8               |
 |  | 1   | Cart Velocity         | -Inf                | Inf               |
 |  | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
 |  | 3   | Pole Angular Velocity | -Inf                | Inf               |
 |
 |  **Note:** While the ranges above denote the possible values for observation space of each element,
 |      it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
 |  -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
 |     if the cart leaves the `(-2.4, 2.4)` range.
 |  -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
 |     if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
 |
 |  ## Rewards
 |
 |  Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
 |  including the termination step, is allotted. The threshold for rewards is 475 for v1.
 |
 |  ## Starting State
 |
 |  All observations are assigned a uniformly random value in `(-0.05, 0.05)`
 |
 |  ## Episode End
 |
 |  The episode ends if any one of the following occurs:
 |
 |  1. Termination: Pole Angle is greater than ±12°
 |  2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)

 |  3. Truncation: Episode length is greater than 500 (200 for v0)
 |
 |  ## Arguments
 |
 |  ```python
 |  import gymnasium as gym
 |  gym.make('CartPole-v1')
 |  ```
 |
 |  On reset, the `options` parameter allows the user to change the bounds used to determine
 |  the new random state.
 |
 |  Method resolution order:
 |      CartPoleEnv
 |      gymnasium.core.Env
 |      typing.Generic
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self, render_mode: Optional[str] = None)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  close(self)
 |      After the user has finished using the environment, close contains the code necessary to "clean up"
the environment.
 |
 |      This is critical for closing rendering windows, database or HTTP connections.
 |
 |  render(self)
 |      Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.
 |
 |      The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
 |      ways to implement the render modes. In addition, list versions for most render modes is achieved through
 |      `gymnasium.make` which automatically applies a wrapper to collect rendered frames.
 |
 |      Note:
 |          As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
 |          should be initialised in ``__init__``.
 |
 |      By convention, if the :attr:`render_mode` is:
 |
 |      - None (default): no render is computed.
 |      - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
 |        This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called. Returns ``None``.
 |      - "rgb_array": Return a single frame representing the current state of the environment.
 |        A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel
image.
 |      - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
 |        for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
 |      - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human)
through the
 |        wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
 |        The frames collected are popped after :meth:`render` is called or :meth:`reset`.
 |
 |      Note:
 |          Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.
 |
 |      .. versionchanged:: 0.25.0
 |
 |          The render function was changed to no longer accept parameters, rather these parameters should
be specified
 |          in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
 |
 |  reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None)
 |      Resets the environment to an initial internal state, returning an initial observation and info.
 |
 |      This method generates a new starting state often with some randomness to ensure that the agent explores the
 |      state space and learns a generalised policy about the environment. This randomness can be controlled
 |      with the ``seed`` parameter otherwise if the environment already has a random number generator and
 |      :meth:`reset` is called with ``seed=None``, the RNG is not reset.
 |
 |      Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.
 |
 |      For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
 |      the seeding correctly.
 |
 |      .. versionchanged:: v0.25
 |
 |          The ``return_info`` parameter was removed and now info is expected to be returned.
 |
 |      Args:
 |          seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
 |              If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
 |              a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
 |              However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
 |              If you pass an integer, the PRNG will be reset even if it already exists.
 |              Usually, you want to pass an integer *right after the environment has been initialized and
then never again*.
 |              Please refer to the minimal example above to see this paradigm in action.
 |          options (optional dict): Additional information to specify how the environment is reset (optional,
 |              depending on the specific environment)
 |
 |      Returns:
 |          observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
 |              (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
 |          info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
 |              the ``info`` returned by :meth:`step`.
 |
 |  step(self, action)
 |      Run one timestep of the environment's dynamics using the agent actions.
 |
 |      When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
 |      reset this environment's state for the next episode.
 |
 |      .. versionchanged:: 0.26
 |
 |          The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
 |          to users when the environment had terminated or truncated which is critical for reinforcement learning
 |          bootstrapping algorithms.
 |
 |      Args:
 |          action (ActType): an action provided by the agent to update the environment state.
 |
 |      Returns:
 |          observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
 |              An example is a numpy array containing the positions and velocities of the pole in CartPole.
 |          reward (SupportsFloat): The reward as a result of taking the action.
 |          terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
 |              which can be positive or negative. An example is reaching the goal state or moving into the lava from
 |              the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
 |          truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
 |              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
 |              Can be used to end the episode prematurely before a terminal state is reached.
 |              If true, the user needs to call :meth:`reset`.
 |          info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
 |              This might, for instance, contain: metrics that describe the agent's performance state, variables that are
 |              hidden from observations, or individual reward terms that are combined to produce the total reward.
 |              In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
 |              however this is deprecated in favour of returning terminated and truncated variables.
 |          done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
 |              return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
 |              A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
 |              a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {}
 |
 |  __orig_bases__ = (gymnasium.core.Env[numpy.ndarray, typing.Union[int, ...
 |
 |  __parameters__ = ()
 |
 |  metadata = {'render_fps': 50, 'render_modes': ['human', 'rgb_array']}
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from gymnasium.core.Env:
 |
 |  __enter__(self)
 |      Support with-statement for the environment.
 |
 |  __exit__(self, *args: 'Any')
 |      Support with-statement for the environment and closes the environment.
 |
 |  __str__(self)
 |      Returns a string of the environment with :attr:`spec` id's if :attr:`spec.
 |
 |      Returns:
 |          A string identifying the environment
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from gymnasium.core.Env:
 |
 |  unwrapped
 |      Returns the base non-wrapped environment.
 |
 |      Returns:
 |          Env: The base non-wrapped :class:`gymnasium.Env` instance
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from gymnasium.core.Env:
 |
 |  __dict__
 |      dictionary for instance variables (if defined)
 |
 |  __weakref__
 |      list of weak references to the object (if defined)
 |
 |  np_random
 |      Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.
 |
 |      Returns:
 |          Instances of `np.random.Generator`
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from gymnasium.core.Env:
 |
 |  render_mode = None
 |
 |  reward_range = (-inf, inf)
 |
 |  spec = None
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from typing.Generic:
 |
 |  __class_getitem__(params) from builtins.type
 |      Parameterizes a generic class.
 |
 |      At least, parameterizing a generic class is the *main* thing this method
 |      does. For example, for some generic class `Foo`, this is called when we
 |      do `Foo[int]` - there, with `cls=Foo` and `params=int`.
 |
 |      However, note that this method is also called when defining generic
 |      classes in the first place with `class Foo(Generic[T]): ...`.
 |
 |  __init_subclass__(*args, **kwargs) from builtins.type
 |      This method is called when a class is subclassed.
 |
 |      The default implementation does nothing. It may be
 |      overridden to extend subclasses.
 ```
