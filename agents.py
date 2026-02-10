from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal
from pydantic import BaseModel, Field
import re
import subprocess

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# https://docs.python.org/3/library/re.html
def extract(code_input):
    code_pattern = r"```(?P<language>[\w+\-]*)[\s\r\n]+(?P<code>.*?)```"
    match = re.search(code_pattern, code_input, re.DOTALL)
    
    if match:
        lang = match.group("language")
        if (lang == ''):
            lang = None
        
        code = match.group("code")
        
        # strip to clean potential leading whitespace or whitespace after
        code = code.strip()
    
    return lang, code

class State(TypedDict):
    request: str
    plan: str
    research: str
    coding: str
    result: str
    critic_des: str
    critic_exp: str
    summary: str
    status: str
    
    

def gen_plan(state: State):
    """Generate a plan to complete the inputted code request"""
    
    msg = llm.invoke(f"Generate a plan to complete the code request inputted by the user. only write out the plan, do not code the whole program. be sure to include unit tests in your plan. The request is {state['request']}")
    return {"plan": msg.content}


def gen_code(state: State):
    """Generates code following the plan that is given, fixes the error that is given, or adjusts the code according to the critique agent"""
    if (state.get("status") == "error"):
      msg = llm.invoke(f"The previous code was written with an error that is outlined here: {state['result']}. The original user request was {state['request']}. The plan was {state['plan']}. The code generated previous was {state['coding']}. Fix the error that caused this error. Generate the code again fully. **WRAP THE CODE IN ``` (triple backticks).**")
    elif (state.get("critic_des") == "RETHINK"):
      msg = llm.invoke(f"""The previous code was written with a wrong output as shown here: {state['result']}. The original user request was {state['request']}. The plan was {state['plan']}. The code generated previous was {state['coding']}. An explaination of the wrong output is outlined here {state['critic_exp']}
                       Fix the error that caused this wrong output. Generate the code again fully. **WRAP THE CODE IN ``` (triple backticks).**""")
    else:
      msg = llm.invoke(f"Generate code following the plan that is given. Use this plan as a baseline of how to generate the code. The original user request was {state['request']}. The plan is {state['plan']}. **WRAP THE CODE IN ``` (triple backticks).")
    return {'coding': msg.content}

def test_code(state: State):
    lang, code = extract(state["coding"])

    # https://docs.python.org/3/library/subprocess.html
    # https://hub.docker.com/_/python (instances)
    # https://realpython.com/python-subprocess/
    # this exec will take in input code and read it and execute it within python. basically we are executing everything with our code
    cmd = ["docker", "run", "--rm", "-i", "python:3.9-alpine", "python", "-c", "import sys; exec(sys.stdin.read())"]
    
    result = subprocess.run(cmd, input=code, text=True, capture_output=True)
    
    if (result.stderr):
      return {"result": result.stderr, "status": "error"}
    else:
      return {"result": result.stdout, "status": "works"} 

def route_after_test(state: State):
  if (state["status"] == "works"):
    return "critique_code"
  else:
    return "gen_code"

class CriticDecision(BaseModel):
  decision: Literal["WORKS", "RETHINK"] = Field(
    description = "Whether the code works as planned or needs to be coded again."
  )
  explaination: str = Field(description=("A brief reason for the decision"))
  

critic_chain = llm.with_structured_output(CriticDecision)

def critique_code(state: State):
  """Critiques code based on what the code generated ouputted, what the original prompt was, and what the plan was. Also checks the quality of the code."""
  critic_result = critic_chain.invoke(f"""You are an expert code reviewer. Review the following code and output to determine if everything works. If the code works how it is supposed to, output 'WORKS'. If it doesn't, output 'RETHINK'. only output 'RETHINK' if the code is fundamentally broken however. provide a brief explaination as well in the explaination section.
                               The plan is {state['plan']}
                               The code is {state['coding']}
                               The current output is {state['result']}""")

  return {"critic_des": critic_result.decision, "critic_exp": critic_result.explaination}

def route_after_critic(state: State):
  if (state["critic_des"] == "RETHINK"):
    return "gen_code"
  else: 
    return "summarize"

def summarize(state: State):
  """Summarizes all of the steps and what happened"""
  summary = llm.invoke(f"""You are an expert code and planning summarizer. Summarize the steps to achieving this final code. Here was the plan: {state['plan']}
                       Here was the code: {state['coding']}
                       Here was the code output: {state['result']}""")
  return {"summary": summary}

workflow = StateGraph(State)

workflow.add_node("gen_plan", gen_plan)
workflow.add_node("gen_code", gen_code)
workflow.add_node("test_code", test_code)
workflow.add_node("critique_code", critique_code)
workflow.add_node("summarize", summarize)

workflow.add_edge(START, "gen_plan")

workflow.add_edge("gen_plan", "gen_code")
workflow.add_edge("gen_code", "test_code")
workflow.add_conditional_edges(
  "test_code",
  route_after_test,
  {"critique_code": "critique_code", "gen_code": "gen_code"},
)
workflow.add_conditional_edges(
  "critique_code",
  route_after_critic,
  {"gen_code": "gen_code", "summarize": "summarize"}
)
workflow.add_edge("summarize", END)

chain = workflow.compile()



state = chain.invoke({"request": """Create a script that solves a Sudoku puzzle. The puzzle should be represented as a 9x9 nested list where 0 is an empty cell. The script must:
1. Include a function is_valid(board, row, col, num) to check if a move is legal.
2. Use a recursive backtracking function to find the solution.
3. Print the final board in a pretty format using pipe | and dash - characters for the grid lines.""", "status": "works"})

print(state["plan"])
print("--------------------------------------------------------\n\n")
print(state["coding"])
print("--------------------------------------------------------\n\n")
print(state["result"])
print("--------------------------------------------------------\n\n")
print(state["critic_des"])
print("--------------------------------------------------------\n\n")
print(state["critic_exp"])
print("--------------------------------------------------------\n\n")
print(state["summary"])
