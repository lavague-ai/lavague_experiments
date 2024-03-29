from lavague.prompts import GEMMA_PROMPT
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"

class LLM(HuggingFaceLLM):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

        model_kwargs = {
            "temperature": 0.0,
        }
        max_new_tokens = 1024

        # We will stop generation as soon as the model outputs the end of Markdown to make inference faster
        stop_token_id = [tokenizer.convert_tokens_to_ids("---"), tokenizer.convert_tokens_to_ids("```")]

        super().__init__(
            model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            stopping_ids=stop_token_id, model_kwargs=model_kwargs
        )

# We needed to steer the model more with a more explicit prompt so we use a custom prompt
prompt_template = GEMMA_PROMPT

# The cleaning is also different in this case
cleaning_function = lambda code: code.split("```")[0]