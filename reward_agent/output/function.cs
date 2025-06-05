public void GiveReward()
{
    // Reward for scoring a goal
    if (team == Team.Blue && ball.transform.position.x > 0) // Blue team scores
    {
        AddReward(1.0f);
    }
    else if (team == Team.Purple && ball.transform.position.x < 0) // Purple team scores
    {
        AddReward(1.0f);
    }
    else
    {
        AddReward(-0.1f); // Penalty for not scoring
    }

    // Reward for being close to the goal
    if (team == Team.Blue && ball.transform.position.x > 0)
    {
        AddReward(0.05f * Mathf.Pow(Mathf.Abs(m_distToGoal), -1));
    }
    else if (team == Team.Purple && ball.transform.position.x < 0)
    {
        AddReward(0.05f * Mathf.Pow(Mathf.Abs(m_distToGoal), -1));
    }

    // Reward for being close to the ball
    if (team == Team.Blue && ball.transform.position.x > 0)
    {
        AddReward(0.05f * Mathf.Pow(Mathf.Abs(m_distBallToGoal), -1));
    }
    else if (team == Team.Purple && ball.transform.position.x < 0)
    {
        AddReward(0.05f * Mathf.Pow(Mathf.Abs(m_distBallToGoal), -1));
    }

    // Reward for kicking the ball
    if (m_kickedBall)
    {
        AddReward(0.01f);
    }

    // Reward for avoiding collisions with walls
    if (!Physics.Raycast(transform.position, transform.forward, out RaycastHit hit, 10f))
    {
        AddReward(0.01f);
    }

    // Reward for staying alive
    AddReward(m_Existential);
}