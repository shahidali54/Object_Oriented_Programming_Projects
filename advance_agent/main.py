import os
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled 
from agents.tool import function_tool
import chainlit as cl

# Disable tracing
set_tracing_disabled(disabled=True)

load_dotenv()

# 🎯 OOP Class for API Setup (Encapsulation)
class GeminiProvider:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    def get_provider(self):
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

# Instantiate GeminiProvider and create model
provider_instance = GeminiProvider()  # 🧱 Object
provider = provider_instance.get_provider()
model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)


# (Abstraction + Static)
class ShahidDataFetcher:
    @staticmethod
    @function_tool("get_shahid_data")
    def get_shahid_data() -> str:
        try:
            response = requests.get("https://github.com/shahidali54")
            if response.status_code == 200:
                return response.text 
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
        
class ShahidAgentFactory:
    @staticmethod
    def create_agent():
        return Agent(
            name="Greeting Agent",
            instructions="""
                You are a Greeting Agent designed to provide friendly interactions and information about Shahid Ali.

                Your responsibilities:
                1. Greet users warmly when they say hello (respond with 'Salam from Shahid Ali')
                2. Say goodbye appropriately when users leave (respond with 'Allah Hafiz from Shahid Ali')
                3. When users request information about Shahid Ali, use the get_shahid_data tool to retrieve and share his profile information
                4. For any questions not related to greetings or Shahid Ali, politely explain: 'I'm only able to provide greetings and information about Shahid Ali. I can't answer other questions at this time.'

                Always maintain a friendly, professional tone and ensure responses are helpful within your defined scope.
            """,
            model=model,
            tools=[ShahidDataFetcher.get_shahid_data]
        )

# Create agent from factory
agent = ShahidAgentFactory.create_agent()



@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Hello! How can I help you today?"
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    
    result = await cl.make_async(Runner.run_sync)(agent, input=history)
    response_text = result.final_output

    await cl.Message(content=response_text).send()

    history.append({"role": "assistant", "content": response_text})
    cl.user_session.set("history", history)