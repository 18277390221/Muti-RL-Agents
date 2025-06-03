import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer, AutoModel 
from PIL import Image
from qwen_vl import QwenVL
import json
import time
import os   
from init_code import code


if __name__ == "__main__":

    image_path = "/home/sda/ZY/Muti-RL-Agents/reward_agent/init_image/init.png" #游戏环境的初始图，这里分为两种，一种是俯视图，一种是第三人称立体图，最终测试哪种效果最好。

    with open('prompt.json', 'r', encoding='utf-8') as file:
        prompts = json.load(file)

    qwen_prompt = prompts.get('qwen_prompt')

    task = """
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

            The following is the source code. You need to provide the specific code of the GiveReward() function.No textual description is required. Please directly provide the code in public void GiveReward() and do not give any other irrelevant code.
        """

    model = QwenVL()
    qwen_prompt = qwen_prompt + task + "\n" + code
    qwen_response = model.chat(qwen_prompt, image_path)

    history = [{"text": qwen_prompt, "response": qwen_response}]
    
    timestamp = int(time.time())  # 使用当前时间戳
    file_name = f"history_{timestamp}.json"
    file_path = os.path.join("/home/sda/ZY/Muti-RL-Agents/reward_agent/history", file_name)  # 保存每一次agent的回答

    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
