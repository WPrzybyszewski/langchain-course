from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from pydantic import BaseModel, Field

load_dotenv()

MAX_ITERATIONS = 10
MODEL_PROVIDER = "openai"
MODEL = "gpt-5.2"

PRICES = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
DISCOUNT_PERCENTAGES = {"bronze": 5, "silver": 12, "gold": 23}


# --- Tool Schemas ---


class ProductPriceInput(BaseModel):
    product: Literal["laptop", "headphones", "keyboard"] = Field(
        description="Exact catalog product name from the allowed list."
    )


class DiscountInput(BaseModel):
    price: float = Field(description="Exact price returned by get_product_price.")
    discount_tier: Literal["bronze", "silver", "gold"] = Field(
        description="Exact discount tier."
    )


# --- Tools (LangChain @tool decorator) ---


@tool(args_schema=ProductPriceInput)
def get_product_price(product: str) -> float:
    """Get the catalog price for one of these exact products: laptop, headphones, keyboard."""
    print(f"    >> Executing get_product_price(product='{product}')")
    return PRICES[product]


@tool(args_schema=DiscountInput)
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply one of these exact discount tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount = DISCOUNT_PERCENTAGES[discount_tier]
    return round(price * (1 - discount / 100), 2)


SYSTEM_PROMPT = (
    "You are a helpful shopping assistant with access to two tools: "
    "get_product_price and apply_discount.\n\n"
    "Catalog products are exactly: laptop, headphones, keyboard.\n"
    "Discount tiers are exactly: bronze, silver, gold.\n\n"
    "STRICT RULES:\n"
    "1. NEVER guess a price. Always call get_product_price first.\n"
    "2. If the user says one of the exact product names above, call get_product_price "
    "immediately. Do not ask for brand, model, or any extra product detail.\n"
    "3. If the user says one of the exact discount tiers above, use it directly. "
    "Do not ask for confirmation.\n"
    "4. Only call apply_discount after you have received a price from "
    "get_product_price, and pass that exact returned price.\n"
    "5. Never calculate the discount yourself.\n"
    "6. Ask a clarification question only if the product or discount tier is truly missing "
    "or not one of the allowed values."
)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(MODEL, model_provider=MODEL_PROVIDER, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            final_text = ai_message.content
            print(f"\nFinal Answer: {final_text}")
            return final_text

        # Force one tool execution per loop iteration so the trace is easy to follow.
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(f"\nReturned Value: {result}")