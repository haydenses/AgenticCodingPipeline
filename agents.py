from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

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

# bench imports
import pandas as pd
import plotly.express as px
import json

# research imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["TRANSFORMERS_VERBOSITY"] = "error"

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# this code is mainly just 4 the streamlit cloud when some1 needs to load their api key w/o a .env file

st.set_page_config(layout="wide")

# code for benchmark
llm = None

if not os.getenv("IS_BENCHMARK") == "true":
  if "configured" not in st.session_state:
    st.session_state.configured = False
    
  
  if "model_provider" not in st.session_state:
    st.session_state.model_provider = "Gemini"
  if "current_model" not in st.session_state:
      st.session_state.current_model = ""
  if "api_key" not in st.session_state:
    st.session_state.api_key = ""

  with st.sidebar:
    st.header("config")
    
    # select model
    temp_model_provider = st.selectbox("select provider", ["Gemini", "Claude", "OpenRouter", "Local-Ollama"])
    
    if temp_model_provider == "Gemini":
      temp_api_key = st.text_input("enter gemini api key:", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
      temp_model_name = st.text_input("model id (e.g. gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview)")
    
    elif temp_model_provider  == "Claude":
      temp_api_key = st.text_input("enter anthropic api key:", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
      temp_model_name = st.text_input("model id (e.g. claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001)")
    
    elif temp_model_provider == "OpenRouter":
      temp_api_key = st.text_input("enter openrouter api key:", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
      temp_model_name = st.text_input("model id (e.g. openai/gpt-5.4, openai/gpt-oss-120b, openai/gpt-5-mini, openai/gpt-5.3-codex)")

    elif temp_model_provider == "Local-Ollama":
      temp_api_key = st.text_input("enter openrouter api key:", type="password")
      temp_model_name = st.text_input("model id (llama-coder:latest, llama3.1:8b)")

    if st.button("submit config"):
      st.session_state.model_provider = temp_model_provider
      st.session_state.api_key = temp_api_key
      st.session_state.model_name = temp_model_name
      st.session_state.configured = True
      st.success(f"using {temp_model_name}")
      st.rerun()
    
  if not st.session_state.configured:
    st.sidebar.write("please configure settings and click submit config")
    st.stop()

  model_provider = st.session_state.model_provider
  model_name = st.session_state.model_name
  api_key = st.session_state.api_key
  
  if model_provider == "Gemini" and api_key:
      os.environ["GOOGLE_API_KEY"] = api_key
      llm = ChatGoogleGenerativeAI(model=model_name)
  elif model_provider == "Claude" and api_key:
      os.environ["ANTHROPIC_API_KEY"] = api_key
      llm = ChatAnthropic(model=model_name)
  elif model_provider == "OpenRouter" and api_key:
      llm = ChatOpenAI(
          model=model_name, 
          api_key=api_key, 
          base_url="https://openrouter.ai/api/v1" # need base url to make sure we are working with openrouter
      )
  elif model_provider == "Local-Ollama":
    llm = ChatOpenAI(
          model=model_name,
          base_url="http://localhost:11434/v1", # The Ollama port
          api_key="ollama" # Dummy key
      )
  else:
      st.write("missing something in the config")
      st.stop()

# this helper function is when the output is a list. gemini 3+ uses outputs as lists and i think other llms do this as well
def parse_content(content):
  if isinstance(content, list):
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and "text" in block)
  return str(content)


# https://docs.python.org/3/library/re.html
def extract(code_input):
    # extra guardrail to extract code just in case ya know
    code_input = parse_content(code_input)
  
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
    
    # research mode
    use_research: bool
    research_decision: str  # will be used for structured outoput and to know if we need to use the web, python docs, or  both
    web_research: str 
    doc_research: str
    
    #learning
    learn_plan: str
    learn_code: str
    learn_summ: str
    
    # test input for benchmark
    test_input: str

# structured output for research decision whether to use web or docs or both
class ResearchDecision(BaseModel):
  decision: Literal["WEB", "DOCS", "BOTH"] = Field(
    description="Whether to search the web for up-to-date/general info or even other library research, python docs for core syntax/innate library info, or both."
  )

def determine_research(state: State):
  """Decides which research tool to use based on the request"""
  decision_chain = llm.with_structured_output(ResearchDecision)
  res = decision_chain.invoke(
    f"The user wants to code: {state['request']}. Does this require up-to-date internet knowledge/third-party libraries (WEB), core offline python documentation/python standard libraries (DOCS), or both (BOTH)?"
  )
  return {"research_decision": res.decision}

def web_search(state: State):
  """Searches the internet with Tavily"""
  # max 3 results because thats all we really need for this project and to not make the context window too big
  tool = TavilySearchResults(max_results=3) 
  docs = tool.invoke({"query": f"Python implementation and best practices for: {state['request']}"})
  
  # format results so that we can see the content and source from the retrival 
  formatted = "\n".join([f"- {d['content']} (Source: {d['url']})" for d in docs])
  return {"web_research": f"WEB SEARCH RESULTS:\n{formatted}"}



def rag_search(state: State):
  """Searches local ChromaDB of python documentation with RAG"""
  # init embeddings to compare embeddings to rag db and connect to chroma db
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
  
  # return top 4 chunks
  docs = vectorstore.similarity_search(state["request"], k=4) 
  
  # format the db content so we can clearly see where we are getting this from. metadata makes the return more rich in context so we can clearly see where we got it from and what relevance it has
  formatted = "\n".join([f"- {d.page_content} (Folder context: {d.metadata.get('category', 'unknown')})" for d in docs])
  return {"doc_research": f"LOCAL DOCS RESULTS:\n{formatted}"}


# compile both
def compile_research(state: State):
    """Compiles and summarizes findings from web, docs, or both"""
    # grab the web and doc research and compile it into same string to give the llm to summarize so that the gen plan step is actually able to use the information without flooding the context window
    raw_research = f"{state.get('web_research', '')}\n\n{state.get('doc_research', '')}"
    
    # summarization step
    msg = llm.invoke(
        f"Summarize the following research into a clear 'Research Brief' for a developer to use while coding the request: {state['request']}.\n\nResearch:\n{raw_research}"
    )
     
    return {"research": parse_content(msg.content)}



def gen_plan(state: State):
    """Generate a plan to complete the inputted code request"""
    
    research_context=""
    # check if research
    if state.get("research"):
      research_context = f"\n\n**CONTEXT FROM RESEARCH AGENT:**\nUse these research findings to inform your plan:\n{state['research']}\n\n"
    
    # this is the normal one
    msg = llm.invoke(f"""Generate a plan to complete the code request inputted by the user. only write out the plan, do not code the whole program. be sure to include unit tests in your plan (**include something to say not to use the unittest library. 
    instead, have the program output expected and outputted behavior using print. do not ask to input any numbers because this script will be run autonomously without any human supervision.**). 
    Research pulled is: {research_context}
    The request is {state['request']}""")
    
    # this is the message for benchmarks. be sure to edit env as well:
    # msg = llm.invoke(f"generate a plan to complete the code request inputted by the user. only write out the plan, do not code the whole program. the program **must** read inputs from standard input using input() or sys.stdin.read() and output the result using print(). do not prompt the user with text like 'enter a number', just read the raw input. the request is {state['request']}")
    
    plan = parse_content(msg.content)
    
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
    
    coding = parse_content(msg.content)
    
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
    
    
    # benchmark code
    # temp file for executions
    
    # uuid_to_use = str(uuid.uuid4()).replace("-","")
    # temp_file = f"temp_script_{uuid_to_use}.py"
    # with open(temp_file, "w", encoding="utf-8") as f:
    #   # write the code to execute to the temp file
    #   f.write(code)
    
    # cmd = [sys.executable, temp_file]
    
    # have to use try and except for unittest library
    try:
      
      test_input = state.get("test_input", "")
      
      result = subprocess.run(cmd, input=test_input, text=True, capture_output=True, timeout=10) #timeout for recursion errors
      
      output = f"stdout: {result.stdout}; \n stderr: {result.stderr}"
      
      if result.returncode == 0:
        status = "works"
      else: 
        status = "error"
      
    except subprocess.TimeoutExpired:
        output = "error: exec timed out (infinite loop or bad code)"
        status = "error"
    
    except Exception as e:
      output = str(e)
      status = "error"
    # finally for temp file bench
    # finally:
    #   if os.path.exists(temp_file):
    #     # cleanup temp file
    #     os.remove(temp_file)

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

def critique_code(state: State):
  """Critiques code based on what the code generated ouputted, what the original prompt was, and what the plan was. Also checks the quality of the code."""
  
  critic_chain = llm.with_structured_output(CriticDecision)
  
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
  return {"summary": parse_content(summary.content)}

# learning route
def learning_route(state: State):
  if (state["mode"] == "learning"):
    return "learning_plan"
  else:
    return END

# learning plan
def learning_plan(state: State):
  """Asks for the user's high-level plan, evaluates it, shows the real plan, and assigns 3 coding tasks."""
  
  
  human_plan_guess = interrupt("before starting, what do you think should be a high level overview of the steps to build this?")
  
  learn_plan = llm.invoke(f"""You are an expert, supportive coding teacher. 
                          The user requested to code this: {state["request"]}
                          You have already generated a 'ground truth' plan behind the scenes: {state["plan"]}
                          
                          The user was asked to guess the high-level plan and they answered: {human_plan_guess}
                          
                          Output a response that includes:
                          1. An evaluation of their high-level plan. Tell them what they got right and what they missed compared to the actual plan.
                          2. Reveal the actual plan to them clearly so they know the exact steps.
                          3. Based on the plan, ask them to write the code for 3 specific functions, features, or important lines of code. (Do **NOT** give them the code answers yet)""")

  return {"learn_plan": parse_content(learn_plan.content)}

# learning code
def learning_code(state: State):
  """Evaluates the user's code, reveals the full code with line-by-line explanations, and asks conceptual questions."""
  
  human_code_snippets = interrupt("enter your code for the 3 features/functions asked previously:")
  
  learn_code = llm.invoke(f"""You are an expert, supportive coding teacher. 
                          In the last step, you asked the user to write code for 3 specific features: {state["learn_plan"]}
                          The user provided this code: {human_code_snippets}
                          You have the 'ground truth' final code behind the scenes: {state["coding"]}
                          
                          Output a response that includes:
                          1. Evaluate the code they wrote. Give constructive feedback on what is correct and what needs fixing.
                          2. Reveal the full ground truth code. **Provide a detailed, line-by-line (or block-by-block) explanation** of how the full code works so they understand the syntax.
                          3. Ask 3 conceptual "Why" or "What if" questions about the full code to test their deeper understanding (e.g., "Why did we use a dictionary here?", "What would happen if the input was empty?").""")

  return {"learn_code": parse_content(learn_code.content)}

# learning summary - connect code and plan
def learning_summary(state: State):
  """Evaluates conceptual answers and provides a final lesson summary."""
  
  human_concept_answers = interrupt("enter your answers to the 3 conceptual questions:")
  
  learn_summ = llm.invoke(f"""You are an expert, supportive coding teacher. 
                          In the last step, you asked the user 3 conceptual questions: {state["learn_code"]}
                          The user answered: {human_concept_answers}
                          The project they are building is: {state["request"]}
                          
                          Output a response that includes:
                          1. Evaluate their answers to the conceptual questions. Correct any misunderstandings gently.
                          2. Give an overall, encouraging summary of this project, the core concepts used, and how they can apply these concepts to future projects.""")
  
  return {"learn_summ": parse_content(learn_summ.content)}


# research router functions new route start for research
def route_start(state: State):
  if state.get("use_research"):
    return "determine_research"
  return "gen_plan"

def route_research(state: State):
  decision = state.get("research_decision")
  if decision == "WEB":
    return["web_search"]
  elif decision == "DOCS":
    return ["rag_search"]
  else: 
    # execute web and rag together
    return["web_search", "rag_search"]




workflow = StateGraph(State)

# research nodes
workflow.add_node("determine_research", determine_research)
workflow.add_node("web_search", web_search)
workflow.add_node("rag_search", rag_search)
workflow.add_node("compile_research", compile_research)

# standard nodes
workflow.add_node("gen_plan", gen_plan)
workflow.add_node("gen_code", gen_code)
workflow.add_node("test_code", test_code)
workflow.add_node("critique_code", critique_code)
workflow.add_node("summarize", summarize)

# learn nodes
workflow.add_node("learning_plan", learning_plan)
workflow.add_node("learning_code", learning_code)
workflow.add_node("learning_summary", learning_summary)

# ------------------------ workflow ---------------------------

# go to route_start router func to see if we skip research or not
workflow.add_conditional_edges(START, route_start)

# determine our research needs if in research mode either going to web or rag research
workflow.add_conditional_edges(
    "determine_research",
    route_research,
    ["web_search", "rag_search"]
)

# converge branches to research (could be both web and rag used so we need this logic)
workflow.add_edge("web_search", "compile_research")
workflow.add_edge("rag_search", "compile_research")

# compile and go to plan
workflow.add_edge("compile_research", "gen_plan")

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

# --------------------------- workflow end --------------------------------

# need a checkpointer for hitl and learning modes. i basically save the memory of the changes that are needed
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
    st.session_state.chain = workflow.compile(checkpointer=st.session_state.checkpointer)

# new chain 
chain = st.session_state.chain

# https://docs.streamlit.io/

auto, learn, hitl, bench = st.tabs(["Autonomous", "Learning", "HITL", "Benchmark Dashboard"])

if not os.getenv("IS_BENCHMARK") == "true":

  with auto:
    if "thread_id" not in st.session_state:
      st.session_state.thread_id = "autonomous1"
    
    input = st.text_area("What do you want to code? (only python supported currently) (autonomous)", height=150, value="Create a simple python script that performs a binary search through a list")
    
    use_research = st.checkbox("Enable Research Agent (search python docs and web auto mode)", value=False)

    if st.button("Run Autonomous Orchestration Loop"):
      # --- creates a mkdwn line separator
      st.write("---")
      input = {"request": input, "iterations": 0, "mode": "autonomous", "use_research": use_research}
        
      # Run the graph and stream outputs step-by-step
      # recursion limit to make it so it doesn't perform a lot of api calls
      for output in chain.stream(input, config={"recursion_limit": 25, "configurable": {"thread_id":st.session_state.thread_id}}):
        for key, value in output.items():
          
          if key == "determine_research":
            with st.expander("step: deciding research strategy", expanded=True):
              st.info(f"elected to use: **{value['research_decision']}**")
          
          elif key == "web_research" or key == "doc_research":
            with st.expander("step: doc and web research", expanded=True):
              st.info(f"doc research gathered: {value['doc_research']} \n\n web research gathered: {value['web_research']}")
          
          elif key == "compile_research":
            with st.expander("step: research compiled", expanded=True):
              st.write(value["research"])
          
          elif key == "gen_plan":
            with st.expander("step: generated plan", expanded=True):
              st.write(f"### Step: **Generated Plan**")
              st.write(value["plan"])
                
          elif key == "gen_code":
            with st.expander("Step: Generated Code", expanded=True):
            # https://docs.streamlit.io/develop/api-reference/text/st.code
              st.code(value["coding"], language="python")
              st.write(f"iteration: {value['iterations']}")
          
          elif key == "test_code":
            with st.expander("step: tested code", expanded=True):
              output_val = value.get("result", "")
              if value["status"] == "works":
                st.success(f"Output: {output_val}")
              else:
                st.error(f"Error: {output_val}")
          
          elif key == "critique_code":
            with st.expander("step: critiqued code", expanded=True):
              st.write(f"Critic Decision: **{value['critic_des']}**")
              st.write(f"Critic Explaination: {value['critic_exp']}")
          
          elif key == "summarize":
            with st.expander("step: summarized code and process", expanded=True):
              st.success("### Final Summary")
              st.write(value["summary"])
          
          
          st.write("---") # Separator line

  # need to work w/ streamlit session states to make sure everything works inside and outside of the loop we are dealing with. saves all the state vars even when we rerun the prompts basically between sessions
  with learn:

    # need to save state for mem and updating the render
    if "learn_thread_id" not in st.session_state:
      st.session_state.learn_thread_id = "learning1"
    # setup learn_hist for streamlit session state
    # learn history basically saves all of the outputs whenever we run the loop and then ask for human inputs. we need to recontruct this history from top to bottom again and save it
    if "learn_history" not in st.session_state:
      st.session_state.learn_history=[]
    
    config_lang = {"recursion_limit": 25, "configurable": {"thread_id":st.session_state.learn_thread_id}}
    
    input = st.text_area("What do you want to code? (only python supported currently) (learning)", height=150, value="Create a simple python script that performs a binary search through a list")
    
    use_research = st.checkbox("Enable Research Agent (search python docs and web learn mode)", value=False)

    left_col, right_col = st.columns(2)
    # i need this all in function to help compartimentalize the outputs easily
    # render learning, then render llm outputs. this is better for the learning module
    def llm_render_learning(output):
      for key, value in output.items():
        if key == "learning_plan":
          with st.expander("step 1: plan evaluation and coding tasks", expanded=True):
            st.write(value["learn_plan"])
        elif key == "learning_code":
          with st.expander("step 2: code evaluation and breakdown", expanded=True):
            st.write(value["learn_code"])
        elif key == "learning_summary":
          with st.expander("step 3: conceptual evaluation and summary", expanded=True):
            st.write(value["learn_summ"])
        elif key == "human_answer":
          with st.expander("your answer: ", expanded=True):
            st.write(f"question: {value['question']}")
            st.write(f"answer: {value['answer']}")
          

    def llm_render_llm(output):
      for key, value in output.items():
        if key == "determine_research":
          with st.expander("behind the scenes: deciding research strategy", expanded=True):
            st.info(f"elected to use: **{value['research_decision']}**")
        
        elif key == "web_research" or key == "doc_research":
          with st.expander("behind the scenes: doc and web research", expanded=True):
            st.info(f"doc research gathered: {value['doc_research']} \n\n web research gathered: {value['web_research']}")
        
        elif key == "compile_research":
          with st.expander("behind the scenes: research compiled", expanded=True):
            st.write(value["research"])
        
        elif key == "gen_plan":
          with st.expander("behind the scenes: generated plan", expanded=False):
            st.write(value["plan"])
        elif key == "gen_code":
          with st.expander("behind the scenes: generated code", expanded=False):
            st.code(value["coding"], language="python")
            st.write(f"iterations: {value['iterations']}")
        elif key == "test_code":
          with st.expander("behind the scenes: tested code", expanded=False):
            if value["status"] == "works":
              st.success(f"Output: {value['result']}")
            else:
              st.error(f"Error: {value['result']}")
        elif key == "critique_code":
          with st.expander("behind the scenes: critiqued code", expanded=False):
            st.write(f"Critic Decision: **{value['critic_des']}**")
            st.write(f"Critic Explaination: {value['critic_exp']}")
        elif key == "summarize":
          with st.expander("behind the scenes: process summary", expanded=False):
            st.write(value["summary"])    
    
    # same button as before
    if st.button("Run Learning Orchestration Loop"):
      # realized that I need uuid here because when we stay on the page (w/o refreshing) we need some type of way to actually start a new session and this does that. 
      # also, cookies can interfere with this even when refreshed so this is just best practice
      st.session_state.thread_id = str(uuid.uuid4())
      config_lang["configurable"]["thread_id"] = st.session_state.learn_thread_id
      
      # reconfigure the learn history
      st.session_state.learn_history =[] 
      
      st.write("---")
      input_chain = {"request": input, "iterations": 0, "mode": "learning", "use_research": use_research}
      
      for output in chain.stream(input_chain, config=config_lang):
        st.session_state.learn_history.append(output)
        llm_render_learning(output)
      st.rerun()
    
    
    
    #render all chunks again
    for chunk in st.session_state.learn_history:
      llm_render_learning(chunk)
      
      
    curr_state = chain.get_state(config_lang)
    
    # https://docs.langchain.com/oss/python/langgraph/interrupts
    
    # this logic is for the learning loop. we basiocally check if there is an interrupt, get that next node that is up (curr_state.next is a bool that returns true if there is a next node (aka if there is an interrupt))
    if curr_state.next:
      next_node = curr_state.next[0]
      
      interrupt_msg = f"waiting for answers before {next_node}"

      st.write(f"{interrupt_msg}")
      
      curr_question = curr_state.tasks[0].interrupts[0].value
      st.write(f"**current question:** {curr_question}")
      
      user_resp = st.text_area("your answers to the questions:", height=200)
      
      if st.button("submit answers and continue the loop"):
        
        st.session_state.learn_history.append({
          "human_answer": {"question": curr_question, "answer": user_resp}
        })
        
        # need command to rerun with the new user response. this is necessary in langgraph documentation essentially
        for output in chain.stream(Command(resume=user_resp), config=config_lang):
          st.session_state.learn_history.append(output)
          llm_render_learning(output)
        
        # rerun streamlit to account for new changes
        st.rerun()
    
    # logic to have the llm answers output. if workflow is finished (no next and there is more than one thing in the session state)
    elif not curr_state.next and len(st.session_state.learn_history) > 0:
      st.markdown("---")
      st.subheader("LLM loop outputs:")
      st.write("Here are the outputs that the llm outputted during its loop:")
      
      for chunk in st.session_state.learn_history:
        llm_render_llm(chunk)
          
  with hitl:
    if "hitl_thread_id" not in st.session_state:
      st.session_state.hitl_thread_id = "hitl1"
    # setup learn_hist for streamlit session state
    # learn history basically saves all of the outputs whenever we run the loop and then ask for human inputs. we need to recontruct this history from top to bottom again and save it
    if "hitl_history" not in st.session_state:
      st.session_state.hitl_history=[]
      
    config_lang = {"recursion_limit": 25, "configurable": {"thread_id":st.session_state.hitl_thread_id}}
    
    input = st.text_area("What do you want to code? (only python supported currently) (hitl)", height=150, value="Create a simple python script that performs a binary search through a list")
    
    use_research = st.checkbox("Enable Research Agent (search python docs and web research mode)", value=False)
    
    def llm_render_outputs(output):
      for key, value in output.items():
        
        if key == "determine_research":
          with st.expander("step: deciding research strategy", expanded=True):
            st.info(f"elected to use: **{value['research_decision']}**")
        
        elif key == "web_research" or key == "doc_research":
          with st.expander("step: doc and web research", expanded=True):
            st.info(f"doc research gathered: {value['doc_research']} \n\n web research gathered: {value['web_research']}")
        
                
        elif key == "compile_research":
          with st.expander("step: research compiled", expanded=True):
            st.write(value["research"])
        
        elif key == "gen_plan":
          with st.expander("step: generated plan", expanded=True):
            # st.write(f"### Step: **Generated Plan**")
            st.write(value["plan"])
            
        elif key == "gen_code":
          with st.expander("Step: Generated Code", expanded=True):
          # https://docs.streamlit.io/develop/api-reference/text/st.code
            st.code(value["coding"], language="python")
            st.write(f"iteration: {value['iterations']}")
        
        elif key == "test_code":
          with st.expander("step: tested code", expanded=True):
            output_val = value.get("result", "")
            if value["status"] == "works":
              st.success(f"Output: {output_val}")
            elif value["status"] == "restart":
              st.code(output_val, language="text")
            else:
              st.error(f"Error: {output_val}")
        
        elif key == "critique_code":
          with st.expander("step: critiqued code", expanded=True):
            st.write(f"Critic Decision: **{value['critic_des']}**")
            st.write(f"Critic Explaination: {value['critic_exp']}")
      
        elif key == "summarize":
          with st.expander("step: summarized code and process", expanded=True):
            st.success("### Final Summary")
            st.write(value["summary"])
    
    if st.button("Run HITL Orchestration Loop"):
    # realized that I need uuid here because when we stay on the page (w/o refreshing) we need some type of way to actually start a new session and this does that. 
    # also, cookies can interfere with this even when refreshed so this is just best practice
      st.session_state.thread_id = str(uuid.uuid4())
      config_lang["configurable"]["thread_id"] = st.session_state.hitl_thread_id
      
      # reconfigure the learn history
      st.session_state.hitl_history =[] 
      
      st.write("---")
      input_chain = {"request": input, "iterations": 0, "mode": "hitl", "use_research": use_research}

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
          
          
# benchmark tab
with bench:
  st.header("Benchmark Results")
    
  try:

    with open("bench_results.json", "r") as f:
      results_data = json.load(f)
      
      
    # get back the data from before. basically new format with model name above, need to get this into one df easily
    data = []
    for name, results in results_data.items():
      for row in results:
        # inject model into data
        row["model"] = name
        data.append(row)
        
    df = pd.DataFrame(data)
    
    total_models = df['model'].nunique()
    total_questions = len(df)
    overall_pass = (df['passed'].sum() / total_questions) * 100 if total_questions > 0 else 0
    
    col1, col2 = st.columns(2)
    col1.metric("Models Tested", total_models)
    col2.metric("Overall Pass Rate", f"{overall_pass:.1f}%")
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    # pass rate chart
    st.subheader("Pass Rate by Model and Difficulty")
    pass_rates = df.groupby(['model', 'difficulty'])['passed'].mean().reset_index()
    pass_rates['passed'] = pass_rates['passed'] * 100
      
    fig = px.bar(
      pass_rates, 
      x="difficulty", 
      y="passed", 
      color="model", 
      barmode="group",
      labels={"passed": "Pass Rate (%)", "difficulty": "Difficulty Level"},
      category_orders={"difficulty": ["easy", "medium", "hard"]}
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
      
    # iterations/time chart
    st.subheader("Efficiency: Iterations vs Time")
    fig2 = px.scatter(
      df, 
      x="iterations_used", 
      y="duration_seconds", 
      color="model",
      symbol="passed",
      hover_data=["difficulty", "problem_id"],
      labels={"iterations_used": "Total Iterations", "duration_seconds": "Duration (Seconds)"}
    )
    st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
    
    # to see the exact chart data
    st.subheader("Raw Data and Judge Reasoning")
    
    u_modles = df["model"].unique().tolist()
    selected_model = st.selectbox(
      "filter by model:",
      options=u_modles
    )
    
    filtered_df = df[df["model"]==selected_model].copy()
    filtered_df = filtered_df.reset_index(drop=True) 
    
    st.write("Click on any row to explore the exact outputs and judge reasoning.")
    
    display_df = filtered_df[['model', 'difficulty', 'problem_id', 'passed', 'output_match', 'iterations_used', 'duration_seconds']]
    
    selected = st.dataframe(
      display_df, 
      width='stretch',
      hide_index=True,
      selection_mode="single-row",
      on_select="rerun"
    )
      

    if selected and len(selected.selection.rows) > 0:
      selected_idx = selected.selection.rows[0]
      row_data = filtered_df.iloc[selected_idx]
      
      st.info(f"**Details for Problem {row_data['problem_id']} ({row_data['model']})**")
      
      col_reason, col_output = st.columns(2)
      with col_reason:
        st.markdown("**Judge Reasoning:**")
        st.write(row_data['judge_reasoning'])
        # only some rows have human revision
        if "human_revision" in row_data and pd.notna(row_data['human_revision']):
          st.warning(f"**Human Revision:** {row_data['human_revision']}")
                
      with col_output:
        st.markdown("**Final Execution Output:**")
        st.code(row_data['final_output'], language="text")
        st.markdown("**Expected Output:**")
        st.code(row_data['expected_output'], language="text")

  except FileNotFoundError:
    st.warning("bench_results.json not found")
