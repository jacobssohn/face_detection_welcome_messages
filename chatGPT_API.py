import openai

class GPTBot:
    def __init__(self, model_engine="text-davinci-003"):
        self.model_engine = model_engine

        with open("api_key.txt") as key:
            openai.api_key = key.readlines()[0]
        self.model_engine = "text-davinci-003"

    def generate_response(self, prompt, max_tokens=1024, temperature=0.5):
        completion = openai.Completion.create(
            engine=self.model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        return completion.choices[0].text



