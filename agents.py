from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


class State(TypedDict):
    request: str
    plan: str
    research: str
    coding: str
    critic: str
    summary: str
    

def gen_plan(state: State):
    """Generate a plan to complete the inputted code request"""
    
    msg = llm.invoke(f"Generate a plan to complete the code request inputted by the user. only write out the plan, do not code the whole program. The request is {state['request']}")
    return {"plan": msg.content}


def gen_code(state: State):
    """Generates code following the plan that is given."""
    
    msg = llm.invoke(f"Generate code following the plan that is given. Use this plan as a baseline of how to generate the code. The original user request was {state['request']}. The plan is {state['plan']}. **WRAP THE CODE IN ``` (triple backticks).")
    return {'coding': msg.content}

# def test_code(state: State):
    

workflow = StateGraph(State)

workflow.add_node("gen_plan", gen_plan)
workflow.add_node("gen_code", gen_code)

workflow.add_edge(START, "gen_plan")

workflow.add_edge("gen_plan", "gen_code")
workflow.add_edge("gen_code", END)

chain = workflow.compile()



state = chain.invoke({"request": "can you build a palindrome checker function in python to check if a word is a palindrome?"})

print(state["plan"])
print(state["coding"])
