from dotenv import load_dotenv
load_dotenv(override=True)

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
)

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for source used by the agent"""
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent"""
    answer:str = Field(description="The answer to the question")
    sources:list[Source] = Field(default_factory=list, description="The sources used to answer the question")




llm = ChatOpenAI(model="gpt-5.2")

@tool
def search(query: str) -> str:
    """Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching the web for: {query}")
    return tavily.search(query)

tools = [TavilySearch()]
agent = create_react_agent(llm, tools, response_format= AgentResponse)

def main():
    print("Starting the agent...")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")]})
    print(result)
    # print("Agent finished")
if __name__ == "__main__":
    main()