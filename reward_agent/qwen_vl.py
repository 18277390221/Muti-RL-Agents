import torch

from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class QwenVL():
    # https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct
    
    def __init__(self, name: str = "Qwen2.5-VL-7B"):
        self.name = name
        
        # if self.name == "Qwen2.5-VL-7B":
        #     model_dir = snapshot_download(
        #         model_id = f"Qwen/{self.name}-Instruct", 
        #         local_files_only = False)
        # else:
        model_dir = "/home_data/home/jinzy2024/ZY/try/Qwen2.5-VL-7B-Instruct"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path = model_dir,
            torch_dtype = torch.bfloat16,
            attn_implementation = "flash_attention_2",
            device_map = "auto")

        # compiler/gcc/7.3.1
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = model_dir, 
            use_fast = True)
        
        self.reset()
    
    
    def reset(self):
        """
        Reset the model history.
        """
        self.img_num = 0
        self.history = []
        torch.cuda.empty_cache()
    
        
    def handle_message(self, prompt_input: str, img_input: str | list = None, is_user: bool = True) -> dict:
        """
        Handle the message in the standard format.
        
        Args:
            prompt_input (str): The text prompt.
            image_input (str | list): The image path or list of image paths.
            is_user (bool): Whether the message is from the user or the assistant.
            
        Format:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file:///path/to/image1.png"},
                        {"type": "image", "image": "file:///path/to/image2.png"},
                        {"type": "text", "text": "What is the difference between these two images?"},
                    ],
                }
            ]
        """
        
        # handle message format
        message = {
            "role": "user" if is_user else "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": prompt_input
                }
            ],
        }
        
        if type(img_input) == str:
            img_input = [img_input]
    
        # handle image input
        if img_input:
            assert is_user, "Image input only support user message."
            for img_path in img_input:
                self.history.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<Image {self.img_num + 1}>: "},
                        {"type": "image",  "image": img_path}
                    ],
                })
                self.img_num += 1

        self.history.append(message)
        return self.history
        
    
    def chat(self, prompt_input: str, img_input: str | list = None) -> str:
        """
        Chat with the model.
        
        Args:
            prompt_input (str): The text prompt.
            image_input (str | list): The image path or list of image paths.
        """
     
        conversation = self.handle_message(prompt_input, img_input, is_user = True)
     
        # pre process
        text = self.processor.apply_chat_template(conversation, tokenize = False, add_generation_prompt = True)
        # print(text)
        img_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(text=[text], images=img_inputs, padding=True, return_tensors="pt").to("cuda")

        # generate the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        self.handle_message(output, is_user = False)
        return output
        
    def to(self, device):
        self.model.to(device)
        return self

if __name__ == "__main__":
    # export PYTHONPATH=$(pwd):$PYTHONPATH
    
    model = QwenVL()

    prompt_input1 = "Can you see the image? Please describe it."
    prompt_input2 = "Can you see the new image? Please describe it."
    prompt_input3 = "Can you remember the first image? Please describe it."
    prompt_input4 = "How many images do you see? Please describe them."
    
    img_path1 = "demo1.png"
    img_path2 = "demo2.png"
    img_path3 = "demo3.png"

    # test with a single image
    model.chat(prompt_input1, img_path1)
    # print(model.history)
    
    # test with history
    model.chat(prompt_input2, img_path2)
    model.chat(prompt_input3)
    # print(model.history)
    
    # test with multiple images
    model.reset()
    model.chat(prompt_input4, [img_path1, img_path2, img_path3])
    # print(model.history)import torch

from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class QwenVL():
    # https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct
    
    def __init__(self, name: str = "Qwen2.5-VL-7B"):
        self.name = name
        
        # if self.name == "Qwen2.5-VL-7B":
        #     model_dir = snapshot_download(
        #         model_id = f"Qwen/{self.name}-Instruct", 
        #         local_files_only = False)
        # else:
        model_dir = "/home_data/home/jinzy2024/ZY/try/Qwen2.5-VL-7B-Instruct"     #本地部署模型路径文件，需要替换

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path = model_dir,
            torch_dtype = torch.bfloat16,
            attn_implementation = "flash_attention_2",
            device_map = "auto")

        # compiler/gcc/7.3.1
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path = model_dir, 
            use_fast = True)
        
        self.reset()
    
    
    def reset(self):
        """
        Reset the model history.
        """
        self.img_num = 0
        self.history = []
        torch.cuda.empty_cache()
    
        
    def handle_message(self, prompt_input: str, img_input: str | list = None, is_user: bool = True) -> dict:
        """
        Handle the message in the standard format.
        
        Args:
            prompt_input (str): The text prompt.
            image_input (str | list): The image path or list of image paths.
            is_user (bool): Whether the message is from the user or the assistant.
            
        Format:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file:///path/to/image1.png"},
                        {"type": "image", "image": "file:///path/to/image2.png"},
                        {"type": "text", "text": "What is the difference between these two images?"},
                    ],
                }
            ]
        """
        
        # handle message format
        message = {
            "role": "user" if is_user else "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": prompt_input
                }
            ],
        }
        
        if type(img_input) == str:
            img_input = [img_input]
    
        # handle image input
        if img_input:
            assert is_user, "Image input only support user message."
            for img_path in img_input:
                self.history.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<Image {self.img_num + 1}>: "},
                        {"type": "image",  "image": img_path}
                    ],
                })
                self.img_num += 1

        self.history.append(message)
        return self.history
        
    
    def chat(self, prompt_input: str, img_input: str | list = None) -> str:
        """
        Chat with the model.
        
        Args:
            prompt_input (str): The text prompt.
            image_input (str | list): The image path or list of image paths.
        """
     
        conversation = self.handle_message(prompt_input, img_input, is_user = True)
     
        # pre process
        text = self.processor.apply_chat_template(conversation, tokenize = False, add_generation_prompt = True)
        # print(text)
        img_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(text=[text], images=img_inputs, padding=True, return_tensors="pt").to("cuda")

        # generate the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        self.handle_message(output, is_user = False)
        return output
        

if __name__ == "__main__":  #主函数仅供测试使用，无需运行
    # export PYTHONPATH=$(pwd):$PYTHONPATH
    
    model = QwenVL()

    prompt_input1 = "Can you see the image? Please describe it."
    prompt_input2 = "Can you see the new image? Please describe it."
    prompt_input3 = "Can you remember the first image? Please describe it."
    prompt_input4 = "How many images do you see? Please describe them."
    
    img_path1 = "demo1.png"
    img_path2 = "demo2.png"
    img_path3 = "demo3.png"

    # test with a single image
    model.chat(prompt_input1, img_path1)
    # print(model.history)
    
    # test with history
    model.chat(prompt_input2, img_path2)
    model.chat(prompt_input3)
    # print(model.history)
    
    # test with multiple images
    model.reset()
    model.chat(prompt_input4, [img_path1, img_path2, img_path3])
    # print(model.history)