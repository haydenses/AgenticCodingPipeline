import json
import time
import os
import re
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import app

load_dotenv()

os.environ["IS_BENCHMARK"] = "true"

# config
DATASET_FILE = "bench.json"
RESULTS_FILE = "bench_results.json"
MAX_QUESTIONS_PER_DIFFICULTY = 5

MODELS_TO_TEST =[
    {"provider": "gemini", "model": "gemini-3-flash-preview"},
    {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
    {"provider": "gemini", "model": "gemini-3.1-pro-preview"}
]

RESEARCH_MODE = [False, True]

# llm as a judge
judge_llm_no_output = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.0)

# structuted output to easily determine if the llm passed or not
class JudgeDecision(BaseModel):
    passed: bool = Field(description="True if the agent's code produced the correct logical output, False otherwise.")
    reasoning: str = Field(description="A brief explanation of why the output is correct or incorrect.")

judge_llm = judge_llm_no_output.with_structured_output(JudgeDecision)

# judge evaluation node. expert competitive because codeparrot/apps is competitive programming problems
def evaluate_with_judge(prompt, generated_code, actual_output, expected_output):
    """Uses the LLM Judge to verify if the actual output matches the expected output."""
    judge_prompt = f"""You are an expert competitive programming judge.
    
--- ORIGINAL PROBLEM ---
{prompt}

--- EXPECTED OUTPUT ---
{expected_output}

--- AGENT'S GENERATED CODE ---
{generated_code}

--- AGENT'S ACTUAL EXECUTION OUTPUT ---
{actual_output}

Task: Your primary goal is to judge the code and the output together to determine if the code actually completes the task and correctly implements the problem's logic. 
- Ensure the code does what it is supposed to do generally, not just for the provided test case (e.g., no hardcoding, logic is sound).
- Check if the actual execution output mathematically and logically matches the expected output.
- Ignore minor whitespace, extra newlines, or print formatting differences.
- If the 'stderr' contains a crash/traceback, it is an automatic fail.
- If the agent output nothing, it is an automatic fail.
"""
    try:
        decision = judge_llm.invoke(judge_prompt)
        return decision.passed, decision.reasoning
    except Exception as e:
        return False, f"judge failed to evaluate: {str(e)}"

# run the benchmarks
def run_benchmarks():
    # open dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # dataset is within the dict called codeparrot apps cause thats were i got it from
    dataset = data.get("codeparrot_apps", {})
    
    # this is so there is no overwritting of previous results. check everything exists and is all good and then store the results so we have them
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}
    
    for model_info in MODELS_TO_TEST:
        provider = model_info["provider"]
        model_name = model_info["model"]
        print(f"\n ----------------  model testing: {model_name} -------------------------- \n")
        
        if provider == "gemini":
            app.llm = ChatGoogleGenerativeAI(model=model_name)
        elif provider == "ollama":
            app.llm = ChatOpenAI(
                model=model_name,
                base_url="http://127.0.0.1:11434/v1", # 127.0.0.1 for wsl
                api_key="ollama"
            )
        
        chain = app.workflow.compile()
        

        all_results[model_name] = []
        
        for difficulty, problems in dataset.items():
            print(f"curr difficulty: {difficulty.upper()}")
            
            # slice list up to max questions per difficulty (5)
            for i, problem in enumerate(problems[:MAX_QUESTIONS_PER_DIFFICULTY]):
                problem_id = problem.get("problem_id", f"{difficulty}_{i}")
                
                # this is when it crashes somtimes to check if its already done
                already_done = any(r.get("problem_id") == problem_id for r in all_results[model_name])
                if already_done:
                    print(f"skipping problem {problem_id}")                
                
                print(f"running problem: {problem_id}...")
                
                # extract inputs and outputs from dataset
                test_input = problem["input_output"]["inputs"][0]
                expected_output = problem["input_output"]["outputs"][0]
                
                start_time = time.time()
                
                try:
                    # run loop
                    final_state = chain.invoke(
                        {
                            "request": problem["question"], 
                            "test_input": test_input, # pipe to stdin
                            "iterations": 0, 
                            "mode": "autonomous",
                            "use_research": False
                        }, 
                        config={"recursion_limit": 25}
                    )
                    # stop timer here
                    
                    duration = time.time() - start_time
                    
                    final_code = final_state.get("coding", "")
                    raw_output = final_state.get("result", "")
                    iterations = final_state.get("iterations", 0)
                    
                    # Call the Judge
                    passed, reasoning = evaluate_with_judge(
                        prompt=problem["question"],
                        generated_code=final_code,
                        actual_output=raw_output,
                        expected_output=expected_output
                    )
                    
                    # regex parse to get the string t osee if it matches
                    stdout_match = re.search(r"stdout:\s*(.*?)\s*;\s*\n\s*stderr:", raw_output, re.DOTALL)
                    # if regex match, get rid of it, else just keep the raw output and we can check later if it worked or not
                    if stdout_match:
                        parsed_output = stdout_match.group(1)
                    else:
                        parsed_output = raw_output
                    
                    # split to get rid of whitespace all together. this caused problems with output matching before
                    output_match = (parsed_output.split() == expected_output.split())
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    print(f"Error executing graph: {e}")
                    final_code = ""
                    raw_output = str(e)
                    final_output = str(e)
                    parsed_output = str(e)
                    iterations = 0
                    passed = False
                    output_match = False
                    reasoning = "execution failed or hit recursion limit"
                
                status = "PASSED" if passed else "FAILED"
                print(f"Result: {status} in {iterations} iterations ({duration:.1f}s)")
                print(f"Output Match: {output_match}")
                print(f"raw Output: {raw_output}")
                print(f"parsed Output: {parsed_output}")
                print(f"expected Output: {expected_output}")
                print(f"Judge Reasoning: {reasoning}\n")
                
                
                # append results
                all_results[model_name].append({
                    "difficulty": difficulty,
                    "problem_id": problem_id,
                    "passed": passed,
                    "output_match": output_match,
                    "iterations_used": iterations,
                    "duration_seconds": round(duration, 2),
                    "judge_reasoning": reasoning,
                    "raw_output": raw_output,
                    "final_output": parsed_output,
                    "expected_output": expected_output
                })
                
                # save to file
                with open(RESULTS_FILE, "w", encoding="utf-8") as out_file:
                    json.dump(all_results, out_file, indent=4)


if __name__ == "__main__":
    run_benchmarks()
