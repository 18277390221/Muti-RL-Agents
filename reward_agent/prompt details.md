#### Zero-Shot Prompt Example
Here is the example of the zero-shot prompt for our tasks:
```
You are an expert in reinforcement learning and code generation. You excel at understanding the objectives of tasks and analyzing potential effects observable from states and actions. Your goal is to write a detailed reward function that efficiently guides agents to learn optimal strategies.

Task:
The environment is a 3v3 soccer game implemented using Unity ML-Agents Toolkit. There are two teams (blue vs. purple), each consisting of two Strikers and one Goalie. The primary goal of each team is to score by getting the ball into the opponent's goal while defending their own goal.

Action Space:
{
  "forward": "Forward", // "Forward" or "Backward"
  "strafe": "Left", //  "Left" or "Right"
  "rotate": "Left" //  "Left" or "Right"
}

State Space:
{
  "ball": {
    "position": [x, y, z],
    "rotation": [qx, qy, qz, qw]
  },
  "goals": {
    "GoalBlue": {"position": [x, y, z]},
    "GoalPurple": {"position": [x, y, z]}
  },
  "agents": [
    {
      "team": "Blue", // "Blue" or "Purple"
      "role": "Striker", // "Striker" or "Goalie"
      "position": [x, y, z],
      "rotation": [qx, qy, qz, qw]
    },
    { /* other agents… */ }
  ]
}

Key Considerations:
1. We already include a goal group reward. A staged reward could make the training more stable.
2. You may define clear, observable metrics that encourage cooperation.
3. You may define role-specific rewards for goalie and striker accordingly.
4. Do not use information not given and focus on most relevant factors.

Present your thought process step-by-step:
1. Analyze and discuss the potential impact of actions and state information on team performance.
2. Suggest detailed reward design ideas.
3. Write the reward function clearly.
```
#### Zero-Shot Reward Functions
We provide zero-shot generated reward functions of our tasks:
```c#
public float ComputeReward(string team, string role, Vector3 agentPos, Vector3 ballPos, Vector3 ownGoalPos, Vector3 oppGoalPos, Vector3 previousBallPos, bool goalScored, bool ownGoalScored, bool kickedBall)
{
    float reward = 0.0f;

    // Global rewards for goals
    if (goalScored)
        reward += 1.0f; // positive reward for scoring
    if (ownGoalScored)
        reward -= 1.0f; // negative reward for conceding

    // Reward for approaching the ball
    float distToBall = Vector3.Distance(agentPos, ballPos);
    float prevDistToBall = Vector3.Distance(agentPos, previousBallPos);
    if (distToBall < prevDistToBall)
        reward += 0.01f; // small incentive to approach ball

    // Reward for actively interacting with the ball
    if (kickedBall)
        reward += 0.1f;

    if (role == "Striker")
    {
        // Reward based on ball moving closer to opponent goal after kick
        float prevGoalDist = Vector3.Distance(previousBallPos, oppGoalPos);
        float currGoalDist = Vector3.Distance(ballPos, oppGoalPos);
        if (currGoalDist < prevGoalDist)
            reward += 0.05f; // incentivize moving ball toward goal
    }
    else if (role == "Goalie")
    {
        // Reward goalie positioning between ball and own goal
        Vector3 ballToGoal = (ownGoalPos - ballPos).normalized;
        Vector3 ballToAgent = (agentPos - ballPos).normalized;
        float positioningAlignment = Vector3.Dot(ballToGoal, ballToAgent);
        
        if (positioningAlignment > 0.8f) // Goalie is effectively positioned
            reward += 0.05f;

        // Reward goalie if they successfully clear the ball away from their goal
        float prevOwnGoalDist = Vector3.Distance(previousBallPos, ownGoalPos);
        float currOwnGoalDist = Vector3.Distance(ballPos, ownGoalPos);
        if (currOwnGoalDist > prevOwnGoalDist && kickedBall)
            reward += 0.1f; // goalie actively defending
    }

    return reward;
}

```

#### Limitations of the Initial Reward Functions
Although the zero-shot rewards are semantically sound, they tend to be practically sub-optimal. In the example above, two issues stand out:
* By rewarding intermediate behaviors too much, the agents may start optimizing for those proxy rewards rather than actual goals.
* If an agent gets incremental rewards that depend on global state, e.g., ball distance to goal, the learning signal becomes noisy or misleading.

#### Human-in-the-Loop Refinement
To mitigate these shortcomings, we solicited targeted human feedback and folded it into the next prompt iteration. This iterative process stabilizes learning and steers the agent toward the real objective while preserving dense, informative feedback.

The refined reward function is:
```c#
public void ApplyAgentReward(string role, Vector3 agentPos, Vector3 ballPos, Vector3 previousBallPos,
                             Vector3 ownGoalPos, Vector3 oppGoalPos, bool kickedBall)
{
    float reward = 0.0f;

    // Small penalty to encourage quick scoring and discourage stalling
    reward -= 0.001f;

    if (kickedBall)
    {
        // Reward meaningful ball interactions (actual kicks)
        reward += 0.02f;

        if (role == "Striker")
        {
            // Reward striker if kick moves ball noticeably closer to opponent's goal
            float prevGoalDist = Vector3.Distance(previousBallPos, oppGoalPos);
            float currGoalDist = Vector3.Distance(ballPos, oppGoalPos);
            
            if (currGoalDist < prevGoalDist - 0.5f) // significant forward progress only
                reward += 0.05f;
        }
        else if (role == "Goalie")
        {
            // Reward goalie clearing the ball from own goal area
            float prevOwnGoalDist = Vector3.Distance(previousBallPos, ownGoalPos);
            float currOwnGoalDist = Vector3.Distance(ballPos, ownGoalPos);

            if (currOwnGoalDist > prevOwnGoalDist + 0.5f) // significant clearance
                reward += 0.05f;
        }
    }

    if (role == "Goalie")
    {
        // Positioning reward activated only when ball is close to goal
        float ballToGoalDist = Vector3.Distance(ballPos, ownGoalPos);
        if (ballToGoalDist < 10.0f) // relevant defensive zone threshold
        {
            Vector3 ballToGoal = (ownGoalPos - ballPos).normalized;
            Vector3 ballToAgent = (agentPos - ballPos).normalized;
            float positioningAlignment = Vector3.Dot(ballToGoal, ballToAgent);

            if (positioningAlignment > 0.8f) // effectively positioned
                reward += 0.02f;
        }
    }

    AddReward(reward);
}


```