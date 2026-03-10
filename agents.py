from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Literal
from pydantic import BaseModel, Field
import re
import subprocess
import streamlit as st
import sys
import os
# import uuid for session ids https://docs.python.org/3/library/uuid.html (uuid4 is just a standard and random one)
import uuid



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# this code is mainly just 4 the streamlit cloud when some1 needs to load their api key w/o a .env file

st.set_page_config(layout="wide")

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
    
    # mode switcher and saves in state what mode we are in
    mode: str
    
    #learning
    learn_plan: str
    learn_code: str
    learn_summ: str
    
    

def gen_plan(state: State):
    """Generate a plan to complete the inputted code request"""
    
    msg = llm.invoke(f"Generate a plan to complete the code request inputted by the user. only write out the plan, do not code the whole program. be sure to include unit tests in your plan (**include something to say not to use the unittest library. instead, have the program output expected and outputted behavior using print. do not ask to input any numbers because this script will be run autonomously without any human supervision.**). The request is {state['request']}")
    
    plan = msg.content
    
    if state.get("mode") == "hitl":
      plan = interrupt({"type": "edit_plan", "draft": plan})
    
    return {"plan": plan, "iterations": 0}


def gen_code(state: State):
    """Generates code following the plan that is given, fixes the error that is given, or adjusts the code according to the critique agent"""
    
    if (state.get("status") == "error"):
      msg = llm.invoke(f"The previous code was written with an error that is outlined here: {state['result']}. The original user request was {state['request']}. The plan was {state['plan']}. The code generated previous was {state['coding']}. Fix the error that caused this error. Generate the code again fully. **WRAP THE CODE IN A SINGLE ``` (triple backticks) BLOCK ONLY ONCE AKA ONLY ONE FILE**")
    elif (state.get("critic_des") == "RETHINK"):
      msg = llm.invoke(f"""The previous code was written with a wrong output as shown here: {state['result']}. The original user request was {state['request']}. The plan was {state['plan']}. The code generated previous was {state['coding']}. An explaination of the wrong output is outlined here {state['critic_exp']}
                       Fix the error that caused this wrong output. Generate the code again fully. **WRAP THE CODE IN A SINGLE ``` (triple backticks) BLOCK ONLY ONCE AKA ONLY ONE FILE**""")
    else:
      msg = llm.invoke(f"Generate code following the plan that is given. Use this plan as a baseline of how to generate the code. The original user request was {state['request']}. The plan is {state['plan']}. **WRAP THE CODE IN A SINGLE ``` (triple backticks) BLOCK ONLY ONCE AKA ONLY ONE FILE**")
    
    coding = msg.content
    
    if state.get("mode") == "hitl":
      # output the draft before so we can edit the coding thing before its submitted, then place back in coding
      coding = interrupt({"draft": coding, "type": "edit_code"})
    
    return {'coding': coding, "iterations": state["iterations"] + 1}

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
        status = "works"
      else: 
        status = "error"
      
      
    except Exception as e:
      output = str(e)
      status = "error"

    if state.get("mode") == "hitl":
      output = interrupt({"draft": output, "type": "test_code"})
      
      # need to check if the output is an instance of dictionary because it will cause an error otherwise
      if isinstance(output, dict) and ((output.get("action")) == "restart"):
        return {"result": output.get("draft", ""), "status": "restart"}
      
    return {"result": output, "status": status}

def route_after_test(state: State):
  #first check for iter, moved the check from gen_code to here because it was causing an infinite loop w/ errors
  if state.get("status") == "restart":
    return "gen_code"
  
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

# learning route
def learning_route(state: State):
  if (state["mode"] == "learning"):
    return "learning_plan"
  else:
    return END

# learning plan
def learning_plan(state: State):
  """Creates a plan for learning what the plan was about. will ask ~5-10 questions about how to create a plan like this, eval answers in next node."""
  learn_plan = llm.invoke(f"""You are a coding teacher given a plan about how to code a project. Output a response that includes 
                          1. How a student should think about creating this plan (ex. steps and guide them to learn how to build this project) 
                          2. 5-10 questions regarding this project to assess them on if they learned anything about how to code this.
                          Their request was {state["request"]}
                          The generated plan to code this project was {state["plan"]}""")

  return {"learn_plan": learn_plan.content}

# learning code
def learning_code(state: State):
  """Creates a plan for learning what the code was about and how to code a project like this. will ask ~5-10 questions about how to create code like this, eval answers in next node."""
  
  # interrupt for human response https://docs.langchain.com/oss/python/langgraph/interrupts
  human_response = interrupt("Enter your answer for the previously asked questions: ")
  
  learn_code = llm.invoke(f"""You are a coding teacher given code on how to code this project. Output a response that includes 
                          1. First, assess their responses to the previous questions from a different output.
                          The questions and learning plan were: {state["learn_plan"]}
                          The human responses were: {human_response}
                          2. How a student should think about creating this code (ex. steps and guide them to learn how to build this project with code) 
                          3. 5-10 questions regarding this project to assess them on if they learned anything about how to code this.
                          Their request was {state["request"]}
                          The generated plan to code this project was {state["plan"]}
                          The generated code was {state["coding"]}""")

  return {"learn_code": learn_code.content}

# learning summary - connect code and plan
def learning_summary(state: State):
  """Assess answers from a student and then outputs a summary for this project and the steps to code it."""
  
  human_response = interrupt("Enter your answer for the previously asked questions: ")
  
  learn_summ = llm.invoke(f"""You are a coding teacher given a plan and code for how to code this project. You will summarize the steps to code this (in a teacher way). Output a response that includes
                          1. First, assess their responses to the previous questions from a different output.
                          The questions and learning code were: {state["learn_code"]}
                          The human responses were: {human_response}
                          2. Give an overall summary of this project and how to code it. (there was already a learning node for generating a plan and learning how to code it)
                          Their request was {state["request"]}
                          The generated plan to code this project was {state["plan"]}
                          The generated code was {state["coding"]}""")
  
  return {"learn_summ": learn_summ.content}

workflow = StateGraph(State)

workflow.add_node("gen_plan", gen_plan)
workflow.add_node("gen_code", gen_code)
workflow.add_node("test_code", test_code)
workflow.add_node("critique_code", critique_code)
workflow.add_node("summarize", summarize)

# learn nodes
workflow.add_node("learning_plan", learning_plan)
workflow.add_node("learning_code", learning_code)
workflow.add_node("learning_summary", learning_summary)

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

# end loop if not learning, continue if learning
workflow.add_conditional_edges(
  "summarize", 
  learning_route,
  {"learning_plan": "learning_plan", END: END},
  )

# learning route
workflow.add_edge("learning_plan", "learning_code")
workflow.add_edge("learning_code", "learning_summary")
workflow.add_edge("learning_summary", END)

# need a checkpointer for hitl and learning modes. i basically save the memory of the changes that are needed
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
    st.session_state.chain = workflow.compile(checkpointer=st.session_state.checkpointer)

# new chain 
chain = st.session_state.chain

# https://docs.streamlit.io/

auto, learn, hitl = st.tabs(["Autonomous", "Learning", "HITL"])

with auto:
  if "thread_id" not in st.session_state:
    st.session_state.thread_id = "autonomous1"
  
  input = st.text_area("What do you want to code? (only python supported currently) (autonomous)", height=150, value="Create a simple python script that performs a binary search through a list.")

  if st.button("Run Autonomous Orchestration Loop"):
    # --- creates a mkdwn line separator
    st.write("---")
    input = {"request": input, "iterations": 0, "mode": "autonomous"}
      
    # Run the graph and stream outputs step-by-step
    # recursion limit to make it so it doesn't perform a lot of api calls
    for output in chain.stream(input, config={"recursion_limit": 25, "configurable": {"thread_id":st.session_state.thread_id}}):
      for key, value in output.items():
              
        if key == "gen_plan":
          with st.expander("Step: Generated Plan", expanded=True):
            st.write(f"### Step: **Generated Plan**")
            st.write(value["plan"])
              
        elif key == "gen_code":
          with st.expander("Step: Generated Code", expanded=True):
          # https://docs.streamlit.io/develop/api-reference/text/st.code
            st.code(value["coding"], language="python")
            st.write(f"iteration: {value['iterations']}")
        
        elif key == "test_code":
          with st.expander("### Step: **Tested Code**", expanded=True):
            output_val = value.get("result", "")
            if value["status"] == "works":
              st.success(f"Output: {output_val}")
            else:
              st.error(f"Error: {output_val}")
        
        elif key == "critique_code":
          with st.expander("Step: Critiqued Code", expanded=True):
            st.write(f"Critic Decision: **{value['critic_des']}**")
            st.write(f"Critic Explaination: {value['critic_exp']}")
        
        elif key == "summarize":
          with st.expander("Step: Summarized Code and Process", expanded=True):
            st.success("### Final Summary")
            st.write(value["summary"])
            
        st.write("---") # Separator line

# need to work w/ streamlit session states to make sure everything works inside and outside of the loop we are dealing with. saves all the state vars even when we rerun the prompts basically between sessions
with learn:

  # need to save state for mem and updating the render
  if "thread_id" not in st.session_state:
    st.session_state.thread_id = "learning1"
  # setup learn_hist for streamlit session state
  # learn history basically saves all of the outputs whenever we run the loop and then ask for human inputs. we need to recontruct this history from top to bottom again and save it
  if "learn_history" not in st.session_state:
    st.session_state.learn_history=[]
  
  config_lang = {"recursion_limit": 25, "configurable": {"thread_id":st.session_state.thread_id}}
  
  input = st.text_area("What do you want to code? (only python supported currently) (learning)", height=150, value="Create a simple python script that performs a binary search through a list.")

  left_col, right_col = st.columns(2)
  # i need this all in function to help compartimentalize the outputs easily
  def llm_render_outputs(output):
    for key, value in output.items():
      if key == "gen_plan":
        with left_col:
          with st.expander("Step: Generated Plan", expanded=True):
            # st.write(f"### Step: **Generated Plan**")
            st.write(value["plan"])
            
      elif key == "gen_code":
        with left_col:
          with st.expander("Step: Generated Code", expanded=True):
          # https://docs.streamlit.io/develop/api-reference/text/st.code
            st.code(value["coding"], language="python")
            st.write(f"iteration: {value['iterations']}")
      
      elif key == "test_code":
        with left_col:
          with st.expander("### Step: **Tested Code**", expanded=True):
            if value["status"] == "works":
              st.success(f"Output: {value['result']}")
            else:
              st.error(f"Error: {value['result']}")
      
      elif key == "critique_code":
        with left_col:
          with st.expander("Step: Critiqued Code", expanded=True):
            st.write(f"Critic Decision: **{value['critic_des']}**")
            st.write(f"Critic Explaination: {value['critic_exp']}")
      
      elif key == "summarize":
        with left_col:
          with st.expander("Step: Summarized Code and Process", expanded=True):
            st.success("### Final Summary")
            st.write(value["summary"])
    
      elif key == "learning_plan":
        with right_col:
          with st.expander("Step: Learning Plan", expanded=True):
            st.write(value["learn_plan"])
      
      elif key == "learning_code":
        with right_col:
          with st.expander("Step: Learning Code", expanded=True):
            st.write(value["learn_code"])
      
      elif key == "learning_summary":
        with right_col:
          with st.expander("Step: Learning Summary", expanded=True):
            st.write(value["learn_summ"])
      
      
  # same button as before
  if st.button("Run Learning Orchestration Loop"):
    # realized that I need uuid here because when we stay on the page (w/o refreshing) we need some type of way to actually start a new session and this does that. 
    # also, cookies can interfere with this even when refreshed so this is just best practice
    st.session_state.thread_id = str(uuid.uuid4())
    config_lang["configurable"]["thread_id"] = st.session_state.thread_id
    
    # reconfigure the learn history
    st.session_state.learn_history =[] 
    
    st.write("---")
    input_chain = {"request": input, "iterations": 0, "mode": "learning"}
    
    for output in chain.stream(input_chain, config=config_lang):
      st.session_state.learn_history.append(output)
      llm_render_outputs(output)
    st.rerun()
  
  
  
  #render all chunks again
  for chunk in st.session_state.learn_history:
    llm_render_outputs(chunk)
    
    
  curr_state = chain.get_state(config_lang)
  
  # https://docs.langchain.com/oss/python/langgraph/interrupts
  
  # this logic is for the learning loop. we basiocally check if there is an interrupt, get that next node that is up (curr_state.next is a bool that returns true if there is a next node (aka if there is an interrupt))
  if curr_state.next:
    next_node = curr_state.next[0]
    
    interrupt_msg = f"waiting for answers before {next_node}"
    with right_col:
      st.write(f"{interrupt_msg}")
      user_resp = st.text_area("Your answers to the questions:")
      
      if st.button("Submit Answers and Continue the Loop"):
        
        # need command to rerun with the new user response. this is necessary in langgraph documentation essentially
        for output in chain.stream(Command(resume=user_resp), config=config_lang):
          st.session_state.learn_history.append(output)
          llm_render_outputs(output)
        
        # rerun streamlit to account for new changes
        st.rerun()
        
with hitl:
  if "thread_id" not in st.session_state:
    st.session_state.thread_id = "hitl1"
  # setup learn_hist for streamlit session state
  # learn history basically saves all of the outputs whenever we run the loop and then ask for human inputs. we need to recontruct this history from top to bottom again and save it
  if "hitl_history" not in st.session_state:
    st.session_state.hitl_history=[]
    
  config_lang = {"recursion_limit": 25, "configurable": {"thread_id":st.session_state.thread_id}}
  
  input = st.text_area("What do you want to code? (only python supported currently) (hitl)", height=150, value="Create a simple python script that performs a binary search through a list.")
  
  def llm_render_outputs(output):
    for key, value in output.items():
      if key == "gen_plan":
        with st.expander("Step: Generated Plan", expanded=True):
          # st.write(f"### Step: **Generated Plan**")
          st.write(value["plan"])
          
      elif key == "gen_code":
        with st.expander("Step: Generated Code", expanded=True):
        # https://docs.streamlit.io/develop/api-reference/text/st.code
          st.code(value["coding"], language="python")
          st.write(f"iteration: {value['iterations']}")
      
      elif key == "test_code":
        with st.expander("### Step: **Tested Code**", expanded=True):
          output_val = value.get("result", "")
          if value["status"] == "works":
            st.success(f"Output: {output_val}")
          elif value["status"] == "restart":
            st.code(output_val, language="text")
          else:
            st.error(f"Error: {output_val}")
      
      elif key == "critique_code":
        with st.expander("Step: Critiqued Code", expanded=True):
          st.write(f"Critic Decision: **{value['critic_des']}**")
          st.write(f"Critic Explaination: {value['critic_exp']}")
    
      elif key == "summarize":
        with st.expander("Step: Summarized Code and Process", expanded=True):
          st.success("### Final Summary")
          st.write(value["summary"])
  
  if st.button("Run HITL Orchestration Loop"):
  # realized that I need uuid here because when we stay on the page (w/o refreshing) we need some type of way to actually start a new session and this does that. 
  # also, cookies can interfere with this even when refreshed so this is just best practice
    st.session_state.thread_id = str(uuid.uuid4())
    config_lang["configurable"]["thread_id"] = st.session_state.thread_id
    
    # reconfigure the learn history
    st.session_state.hitl_history =[] 
    
    st.write("---")
    input_chain = {"request": input, "iterations": 0, "mode": "hitl"}

    for output in chain.stream(input_chain, config=config_lang):
      st.session_state.hitl_history.append(output)
      llm_render_outputs(output)
    st.rerun()

  for chunk in st.session_state.hitl_history:
    llm_render_outputs(chunk)
  
  curr_state = chain.get_state(config_lang)
  
  if curr_state.next:
    next_node = curr_state.next[0]
    
    interrupt_msg = f"waiting for answers before {next_node}"
    
    # get the contents of the current node
    interrupt_contents = curr_state.tasks[0].interrupts[0].value

    # IF TEST CODE, THEN SEE IF WE WANT TO CHANGE PLAN AFTER VIEWING OUTPUT
    if (interrupt_contents["type"] == "test_code"):
      
      # subheaders for visual clarity
      st.subheader("OPTION 1: Continue Loop")
      with st.expander("current output: ", expanded=True):
        st.code(interrupt_contents["draft"], language="text")
      
      if st.button("continue the loop fully",  key = "btn_continue_aft_test"):
        for output in chain.stream(Command(resume=interrupt_contents["draft"]), config=config_lang):
          st.session_state.hitl_history.append(output)
          llm_render_outputs(output)
        st.rerun()
      
      st.write("---")
      
      st.subheader("OPTION 2: edit plan and start from the beginning")
      curr_plan = curr_state.values.get("plan")
      new_plan = st.text_area("edit the plan: ", value=curr_plan, height = 300)
      
      if (st.button("OPTION 2: update plan and restart code gen", key="update_plan")):
        # new command to go to gen code right after rivising plan
        plan_output = {"gen_plan": {"plan": new_plan}}
        command=Command(resume={"action": "restart", "draft": interrupt_contents["draft"]}, update={"plan": new_plan, "iterations": 0})
        for output in chain.stream(command, config=config_lang):
          st.session_state.hitl_history.append(output)
          llm_render_outputs(output)
          if "test_code" in output:
            st.session_state.hitl_history.append(plan_output)
            llm_render_outputs(plan_output)
        st.rerun()
        
      
    else:    
      st.write(f"{interrupt_msg}")
      user_resp = st.text_area("edit the contents: ", value=interrupt_contents["draft"], height=300)
      
      if st.button("continue the hitl loop"):
        
        # need command to rerun with the new user response. this is necessary in langgraph documentation essentially
        for output in chain.stream(Command(resume=user_resp), config=config_lang):
          st.session_state.hitl_history.append(output)
          llm_render_outputs(output)
        
        # rerun streamlit to account for new changes
        st.rerun()

