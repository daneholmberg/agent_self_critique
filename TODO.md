# Completed Tasks

- Allow for the entry point to be ideation. Between N different high temperature LLMs.
  - This would be fed back to the UI for a selection to be made. So I guess this would be a seperate agent?
- Give the video analyzer context that it can only see at N frames per second so it might not see transitions as well.
- we should pass in what should be animated. Then make a specific rubric based on that.

---

# Pending Tasks

- Expand the past mistakes to be a little less concise
- I've also made adjustments to how failure reasons (validation_error from execution, feedback from evaluation) are explicitly passed to the summarize_single_failure_node using the conditional edge mapping feature of LangGraph
- Have a "think to self" node? If we can't get over the hump. To analyze past errors and current rubric. Maybe do this every time and add it to past history.
  - "What did I try, what could i add?"
  - To build up this history better. History of runs is probably going to be key to get good results
  - We could add a new key to the state to keep track of past thoughts. So every run (without an error?) we could have another LLM call after eval.
  - Did we achieve what we set out to do? If not, why not? Add this all to context to have a running history of the models thoughts so it can truly work through the issue.
  - Put these thoughts next to each attempt. Right now we have attempt N: failure reason:. Lets have attempt N:
    1. Summary of what the code is attempting to achieve.
    2. Failure reasons
    3. Thoughts on what it was trying to accomplish.
- pass rubric mod back to UI for approval and suggestion?
- Any mistakes should be added to a file. We can then take this file and generate a token optimized version of it so that we can try to prevent them in the future.