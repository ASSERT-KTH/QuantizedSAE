import anthropic

class AnthropicHandler:
    # def __init__(self, model="claude-3-5-sonnet-20240620"):
    def __init__(self, model="claude-3-haiku-20240307"):
        with open("api/api.txt", "r") as file:
            api_key = file.read().strip()
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def get_response(self, prompt, max_tokens=1024, temperature=0):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_conversation_response(self, conversation_history, max_tokens=1024, temperature=0, system=None):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=conversation_history
            )
            
            return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"