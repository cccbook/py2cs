# chainMath.py

* 來源 -- https://python.langchain.com/en/latest/getting_started/getting_started.html

```
$ python chainMath.py


?[1m> Entering new AgentExecutor chain...?[0m
?[32;1m?[1;3m I need to find the temperature first, then use the calculator to raise it to the .023
power.
Action: Search
Action Input: "High temperature in SF yesterday"?[0m
Observation: ?[36;1m?[1;3mHigh: 57.2ºf @10:55 AM Low: 53.06ºf @4:56 AM Approx. Precipitation / Rain
Total: in. 1hr.?[0m
Thought:?[32;1m?[1;3m I now need to use the calculator to raise 57.2 to the .023 power
Action: Calculator
Action Input: 57.2^.023?[0m
Observation: ?[33;1m?[1;3mAnswer: 1.0975393720872189?[0m
Thought:?[32;1m?[1;3m I now know the final answer
Final Answer: 1.0975393720872189?[0m

?[1m> Finished chain.?[0m
result= 1.0975393720872189
```


## Agents: Dynamically Call Chains Based on User Input

So far the chains we’ve looked at run in a predetermined order.

Agents no longer do: they use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning to the user.

When used correctly agents can be extremely powerful. In this tutorial, we show you how to easily use agents through the simplest, highest level API.

In order to load agents, you should understand the following concepts:

* Tool: A function that performs a specific duty. This can be things like: Google Search, Database lookup, Python REPL, other chains. The interface for a tool is currently a function that is expected to have a string as an input, with a string as an output.

* LLM: The language model powering the agent.

* Agent: The agent to use. This should be a string that references a support agent class. Because this notebook focuses on the simplest, highest level API, this only covers using the standard supported agents. If you want to implement a custom agent, see the documentation for custom agents (coming soon).