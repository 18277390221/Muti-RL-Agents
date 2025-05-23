import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer, AutoModel 
from PIL import Image
from qwen_vl import QwenVL
import json
import time
import os   
from init_code import code


if __name__ == "__main__":

    image_path = "/home/sda/ZY/Muti-RL-Agents/reward_agent/init_image/image.png" #游戏环境的初始图，这里分为两种，一种是俯视图，一种是第三人称立体图，最终测试哪种效果最好。

    with open('prompt.json', 'r', encoding='utf-8') as file:
        prompts = json.load(file)

    qwen_prompt = prompts.get('qwen_prompt')

    task = """
            请给出一个合适的reward奖励策略，要求输出代码到
            public void GiveReward()
            {
                // TODO
            }。
        """


    model = QwenVL()
    qwen_prompt = qwen_prompt + task
    qwen_response = model.chat(qwen_prompt, image_path)

    history = [{"text": qwen_prompt, "response": qwen_response}]
    
    timestamp = int(time.time())  # 使用当前时间戳
    file_name = f"history_{timestamp}.json"
    file_path = os.path.join("/home/sda/ZY/Muti-RL-Agents/reward_agent/history", file_name)  # 保存每一次agent的回答

    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
