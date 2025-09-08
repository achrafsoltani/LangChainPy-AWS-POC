# Ensure your AWS credentials are configured

from langchain.chat_models import init_chat_model

model = init_chat_model("eu.anthropic.claude-3-7-sonnet-20250219-v1:0", model_provider="bedrock_converse")

from langchain_core.messages import HumanMessage

output = model.invoke([HumanMessage(content="Hi! I'm Bob")])
print(output.content)

output = model.invoke([HumanMessage(content="Can you summarize the book 'Fairy tale' by Stephen King")])
print(output.content)