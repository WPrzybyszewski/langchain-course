from dotenv import load_dotenv
load_dotenv(override=True)
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch


llm = ChatOpenAI(model="gpt-5.2")


tools = [TavilySearch()]
agent = create_react_agent(llm, tools)

def main():
    print("Starting the agent...")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")]})
    # print(result)
    # print("Agent finished")
if __name__ == "__main__":
    main()