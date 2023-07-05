from dotenv import load_dotenv
from langchain.llms import OpenAI # Importing the LLM wrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Importing the os module to access environment variables
# The os module is meant to provide a way of using operating system dependent functionality
import os 
openai_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

Calling the LLM on an input
llm = OpenAI(temperature=0.9) # Temperature is a hyperparameter of LLMs that controls the randomness of predictions
text = "What is the name of the oldest cryptocurrency?"
print(llm(text)) # The text is the input to the LLM

# Prompt templates 
# Prompt templates are a way to structure the input to the LLM
prompt = PromptTemplate(
    input_variables=["cryptocurrency"], # The input variables are the variables that will be passed to the LLM
    template="What are the oldest {cryptocurrency}?",
 )

print(prompt.format(cryptocurrency="cryptocurrencies")) # The prompt where cryptocurrency equals cryptocurrency
print(llm(prompt.format(cryptocurrency="cryptocurrencies"))) # The prompt template is fed into the LLM

# Chains are a way to combine LLMs and prompts in multi-step workflows
prompt = PromptTemplate(
    input_variables=["cryptocurrency"],
    template="What are the five {cryptocurrency} with the highest price in USD and what are their prices? Inform the date at end of your answer.",
 )

chain = LLMChain(llm=llm, prompt=prompt) # The LLMChain is initialized with the LLM and the prompt template
print(chain.run("cryptocurrencies")) # The chain is run with the input variable cryptocurrency equal to cryptocurrencies

# Agents: Dynamically call chains based on user input
pip install google-search-results
llm = OpenAI(temperature=0.7) # Load the model
tools = load_tools(["serpapi", "llm-math"], llm=llm) # Load the tools 

# Initialize the agent:
# 1. The tools
# 2. The LLM
# 3. The type of agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True) 
agent.run("Who founded Ethereum? What is the largest prime number that is smaller than the Ethereum price? Include the date in your answer.")

# Memory: Add state to chains and agents
llm = OpenAI(temperature=0.7)

# Initialize the ConversationChain with a memory buffer
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory(), verbose=True)

# Run the conversation
response = conversation.run("Oi, meu nome é Ferdinando. Qual é o seu nome?")
print(response)

response = conversation.run("Sabe me dizer quais são as maiores empresas brasileiras de tecnologia?")
print(response)

response = conversation.run("Qual é o meu nome?")
print(response)
