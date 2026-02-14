from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal
from pydantic import BaseModel, Field
import re
import subprocess
import streamlit as st
import sys
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# this code is mainly just 4 the streamlit cloud when some1 needs to load their api key w/o a .env file

with st.sidebar:
  st.header("config")
  if not api_key:
    api_key = st.text_input("enter gemini api key:", type="password")
    if api_key:
      os.environ["GOOGLE_API_KEY"] = api_key
  else:
    st.success("key loaded")

if not api_key:
  st.write("enter a google api key to use the program.")
  st.stop()
    


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

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
    iterations: int
    
    

def gen_plan(state: State):
    """Generate a plan to complete the inputted code request"""
    
    msg = llm.invoke(f"Generate a plan to complete the code request inputted by the user. only write out the plan, do not code the whole program. be sure to include unit tests in your plan. The request is {state['request']}")
    return {"plan": msg.content, "iterations": 0}


def gen_code(state: State):
    """Generates code following the plan that is given, fixes the error that is given, or adjusts the code according to the critique agent"""
    
    if (state.get("status") == "error"):
      msg = llm.invoke(f"The previous code was written with an error that is outlined here: {state['result']}. The original user request was {state['request']}. The plan was {state['plan']}. The code generated previous was {state['coding']}. Fix the error that caused this error. Generate the code again fully. **WRAP THE CODE IN A SINGLE ``` (triple backticks) BLOCK ONLY ONCE AKA ONLY ONE FILE**")
    elif (state.get("critic_des") == "RETHINK"):
      msg = llm.invoke(f"""The previous code was written with a wrong output as shown here: {state['result']}. The original user request was {state['request']}. The plan was {state['plan']}. The code generated previous was {state['coding']}. An explaination of the wrong output is outlined here {state['critic_exp']}
                       Fix the error that caused this wrong output. Generate the code again fully. **WRAP THE CODE IN A SINGLE ``` (triple backticks) BLOCK ONLY ONCE AKA ONLY ONE FILE**""")
    else:
      msg = llm.invoke(f"Generate code following the plan that is given. Use this plan as a baseline of how to generate the code. The original user request was {state['request']}. The plan is {state['plan']}. **WRAP THE CODE IN A SINGLE ``` (triple backticks) BLOCK ONLY ONCE AKA ONLY ONE FILE**")
    return {'coding': msg.content, "iterations": state["iterations"] + 1}

def test_code(state: State):
    lang, code = extract(state["coding"])

    # https://docs.python.org/3/library/subprocess.html
    # https://hub.docker.com/_/python (instances)
    # https://realpython.com/python-subprocess/
    # this exec will take in input code and read it and execute it within python. basically we are executing everything with our code
    # docker doesn't work w/ streamlit cloud, we can use the sys.exec way only without opening a docker script since streamlit cloud is a container itself 
    # would recommend using docker container if running program locally
    # cmd = ["docker", "run", "--rm", "-i", "python:3.9-alpine", "python", "-c", "import sys; exec(sys.stdin.read())"]
    cmd = [sys.executable, "-c", "import sys; exec(sys.stdin.read())"]
    
    # have to use try and except for unittest library
    try:
        
      result = subprocess.run(cmd, input=code, text=True, capture_output=True, timeout=10) #timeout for recursion errors
      
      output = f"stdout: {result.stdout}; \n stderr: {result.stderr}"
      
      if result.returncode == 0:
        return {"result": output, "status": "works"}
      else: 
        return {"result": output, "status": "error"}
      
      
    except Exception as e:
      return {"result": str(e), "status": "error"}

def route_after_test(state: State):
  #first check for iter, moved the check from gen_code to here because it was causing an infinite loop w/ errors
  if (state["iterations"] >= 3):
    return "critique_code"
  
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
  if (state["iterations"] >= 3):
    return "summarize"
  elif (state["critic_des"] == "RETHINK"):
    return "gen_code"
  else: 
    return "summarize"

def summarize(state: State):
  """Summarizes all of the steps and what happened"""
  summary = llm.invoke(f"""You are an expert code and planning summarizer. Summarize the steps to achieving this final code. Here was the plan: {state['plan']}
                       Here was the code: {state['coding']}
                       Here was the code output: {state['result']}""")
  return {"summary": summary.content}

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

# https://docs.streamlit.io/

input = st.text_area("What do you want to code? (only python supported currently)", height=150, value="Create a python script that performs a binary search through a list.")

if st.button("Run Orchestration Loop"):
  # --- creates a mkdwn line separator
  st.write("---")
  input = {"request": input, "iterations": 0}
    
  # Run the graph and stream outputs step-by-step
  # recursion limit to make it so it doesn't perform a lot of api calls
  for output in chain.stream(input, config={"recursion_limit": 25}):
    for key, value in output.items():
            
      if key == "gen_plan":
        st.write(f"### Step: **Generated Plan**")
        st.write(value["plan"])
            
      elif key == "gen_code":
        st.write(f"### Step: **Generated Code**")
        # https://docs.streamlit.io/develop/api-reference/text/st.code
        st.code(value["coding"], language="python")
        st.write(f"iteration: {value['iterations']}")
      
      elif key == "test_code":
        st.write(f"### Step: **Tested Code**")
        if value["status"] == "works":
          st.success(f"Output: {value['result']}")
        else:
          st.error(f"Error: {value['result']}")
      
      elif key == "critique_code":
        st.write(f"### Step: **Critiqued Code**")
        st.write(f"Critic Decision: **{value['critic_des']}**")
        st.write(f"Critic Explaination: {value['critic_exp']}")
      
      elif key == "summarize":
        st.write(f"### Step: **Summarized the code**")
        st.success("### Final Summary")
        st.write(value["summary"])
          
      st.write("---") # Separator line
