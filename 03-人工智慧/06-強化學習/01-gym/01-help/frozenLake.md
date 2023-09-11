# FrozenLake

* https://www.gymlibrary.dev/environments/toy_text/frozen_lake/

![](https://www.gymlibrary.dev/_images/frozen_lake.gif)

4*4 的格狀世界，用傳回值 `observation = row*4+col` 表示，主角可上下左右移動，只有到達目標才加一分，否則都不加分。


```
$ python env_help.py FrozenLake-v1
Help on FrozenLakeEnv in module gymnasium.envs.toy_text.frozen_lake object:

class FrozenLakeEnv(gymnasium.core.Env)
 |  FrozenLakeEnv(render_mode: Optional[str] = None, desc=None, map_name='4x4', is_slippery=True)
 |
 |  Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
 |  by walking over the frozen lake.
 |  The player may not always move in the intended direction due to the slippery nature of the frozen lake.
 |
 |  ## Description
 |  The game starts with the player at location [0,0] of the frozen lake grid world with the
 |  goal located at far extent of the world e.g. [3,3] for the 4x4 environment.
 |
 |  Holes in the ice are distributed in set locations when using a pre-determined map
 |  or in random locations when a random map is generated.
 |
 |  The player makes moves until they reach the goal or fall in a hole.
 |
 |  The lake is slippery (unless disabled) so the player may move perpendicular
 |  to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).
 |
 |  Randomly generated worlds will always have a path to the goal.
 |
 |  Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
 |  All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).
 |
 |  ## Action Space
 |  The action shape is `(1,)` in the range `{0, 3}` indicating
 |  which direction to move the player.
 |
 |  - 0: Move left
 |  - 1: Move down
 |  - 2: Move right
 |  - 3: Move up
 |
 |  ## Observation Space
 |  The observation is a value representing the player's current position as |  current_row * nrows + current_col (where both the row and col start at 0).
 |
 |  For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
 |  The number of possible observations is dependent on the size of the map. |
 |  The observation is returned as an `int()`.
 |
 |  ## Starting State
 |  The episode starts with the player in state `[0]` (location [0, 0]).
 |
 |  ## Rewards
 |
 |  Reward schedule:
 |  - Reach goal: +1
 |  - Reach hole: 0
 |  - Reach frozen: 0
 |
 |  ## Episode End
 |  The episode ends if the following happens:
 |
 |  - Termination:
 |      1. The player moves into a hole.
 |      2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
 |
 |  - Truncation (when using the time_limit wrapper):
 |      1. The length of the episode is 100 for 4x4 environment, 200 for 8x8 environment.
 |
 |  ## Information
 |
 |  `step()` and `reset()` return a dict with the following keys:
 |  - p - transition probability for the state.
 |
 |  See <a href="#is_slippy">`is_slippery`</a> for transition probability information.
 |
 |
 |  ## Arguments
 |
 |  ```python
 |  import gymnasium as gym
 |  gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
 |  ```
 |
 |  `desc=None`: Used to specify maps non-preloaded maps.
 |
 |  Specify a custom map.
 |  ```
 |      desc=["SFFF", "FHFH", "FFFH", "HFFG"].
 |  ```
 |
 |  A random generated map can be specified by calling the function `generate_random_map`.
 |  ```
 |  from gymnasium.envs.toy_text.frozen_lake import generate_random_map
 |
 |  gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
 |  ```
 |
 |  `map_name="4x4"`: ID to use any of the preloaded maps.
 |  ```
 |      "4x4":[
 |          "SFFF",
 |          "FHFH",
 |          "FFFH",
 |          "HFFG"
 |          ]
 |
 |      "8x8": [
 |          "SFFFFFFF",
 |          "FFFFFFFF",
 |          "FFFHFFFF",
 |          "FFFFFHFF",
 |          "FFFHFFFF",
 |          "FHHFFFHF",
 |          "FHFFHFHF",
 |          "FFFHFFFG",
 |      ]
 |  ```
 |
 |  If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
 |  `None` a random 8x8 map with 80% of locations frozen will be generated.
 |
 |  <a id="is_slippy"></a>`is_slippery=True`: If true the player will move in intended direction with
 |  probability of 1/3 else will move in either perpendicular direction with |  equal probability of 1/3 in both directions.
 |
 |  For example, if action is left and is_slippery is True, then:
 |  - P(move left)=1/3
 |  - P(move up)=1/3
 |  - P(move down)=1/3
 |
 |
 |  ## Version History
 |  * v1: Bug fixes to rewards
 |  * v0: Initial version release
 |
 |  Method resolution order:
 |      FrozenLakeEnv
 |      gymnasium.core.Env
 |      typing.Generic
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self, render_mode: Optional[str] = None, desc=None, map_name='4x4', is_slippery=True)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  close(self)
 |      After the user has finished using the environment, close contains th
e code necessary to "clean up" the environment.
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
 |      - "rgb_array": Return a single frame representing the current state
of the environment.
 |        A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing
RGB values for an x-by-y pixel image.
 |      - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
 |        for each time step. The text can include newlines and ANSI escape
sequences (e.g. for colors).
 |      - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
 |        wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
 |        The frames collected are popped after :meth:`render` is called or
:meth:`reset`.
 |
 |      Note:
 |          Make sure that your class's :attr:`metadata` ``"render_modes"``
key includes the list of supported modes.
 |
 |      .. versionchanged:: 0.25.0
 |
 |          The render function was changed to no longer accept parameters,
rather these parameters should be specified
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
 |      Therefore, :meth:`reset` should (in the typical use case) be called
with a seed right after initialization and then never again.
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
 |              Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
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
 |  step(self, a)
 |      Run one timestep of the environment's dynamics using the agent actions.
 |
 |      When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
 |      reset this environment's state for the next episode.
 |
 |      .. versionchanged:: 0.26
 |
 |          The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
 |          to users when the environment had terminated or truncated which
is critical for reinforcement learning
 |          bootstrapping algorithms.
 |
 |      Args:
 |          action (ActType): an action provided by the agent to update the
environment state.
 |
 |      Returns:
 |          observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
 |              An example is a numpy array containing the positions and velocities of the pole in CartPole.
 |          reward (SupportsFloat): The reward as a result of taking the action.
 |          terminated (bool): Whether the agent reaches the terminal state
(as defined under the MDP of the task)
 |              which can be positive or negative. An example is reaching the goal state or moving into the lava from
 |              the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
 |          truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
 |              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
 |              Can be used to end the episode prematurely before a terminal state is reached.
 |              If true, the user needs to call :meth:`reset`.
 |          info (dict): Contains auxiliary diagnostic information (helpful
for debugging, learning, and logging).
 |              This might, for instance, contain: metrics that describe the agent's performance state, variables that are
 |              hidden from observations, or individual reward terms that are combined to produce the total reward.
 |              In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
 |              however this is deprecated in favour of returning terminated and truncated variables.
 |          done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
 |              return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
 |              A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
 |              a certain timelimit was exceeded, or the physics simulation
has entered an invalid state.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {}
 |
 |  __parameters__ = ()
 |
 |  metadata = {'render_fps': 4, 'render_modes': ['human', 'ansi', 'rgb_ar...
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
 |  __orig_bases__ = (typing.Generic[~ObsType, ~ActType],)
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
