import openai

# Set the API key and model
openai.api_key = "YOUR_API_KEY"
model_engine = "text-davinci-002"

# Define the prompt
prompt = "Hello, how are you today?"

# Generate a response from the model
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the response
print(completion.choices[0].text)