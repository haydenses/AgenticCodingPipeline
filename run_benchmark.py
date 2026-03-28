import json
import time
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import app

load_dotenv()

os.environ["IS_BENCHMARK"] = "true"

# config
DATASET_FILE = "bench.json"
RESULTS_FILE = "bench_results.json"
MAX_QUESTIONS_PER_DIFFICULTY = 5

MODELS_TO_TEST =[
    "gemini-2.5-flash-lite",
]

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

Task: Did the agent's actual execution output mathematically and logically match the expected output? 
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
    all_results =[]
    
    for model_name in MODELS_TO_TEST:
        print(f"model testing: {model_name}")
        
        app.llm = ChatGoogleGenerativeAI(model=model_name)
        chain = app.workflow.compile()
        
        for difficulty, problems in dataset.items():
            print(f"curr difficulty: {difficulty.upper()}")
            
            # slice list up to max questions per difficulty (5)
            for i, problem in enumerate(problems[:MAX_QUESTIONS_PER_DIFFICULTY]):
                problem_id = problem.get("problem_id", f"{difficulty}_{i}")
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
                            "mode": "autonomous"
                        }, 
                        config={"recursion_limit": 25}
                    )
                    
                    final_code = final_state.get("coding", "")
                    final_output = final_state.get("result", "")
                    iterations = final_state.get("iterations", 0)
                    
                    # Call the Judge
                    passed, reasoning = evaluate_with_judge(
                        prompt=problem["question"],
                        generated_code=final_code,
                        actual_output=final_output,
                        expected_output=expected_output
                    )
                    
                except Exception as e:
                    print(f"Error executing graph: {e}")
                    final_code = ""
                    final_output = str(e)
                    iterations = 0
                    passed = False
                    reasoning = "execution failed or hit recursion limit"
                
                duration = time.time() - start_time
                status = "PASSED" if passed else "FAILED"
                print(f"Result: {status} in {iterations} iterations ({duration:.1f}s)")
                print(f"Judge Reasoning: {reasoning}\n")
                
                # append results
                all_results.append({
                    "model": model_name,
                    "difficulty": difficulty,
                    "problem_id": problem_id,
                    "passed": passed,
                    "iterations_used": iterations,
                    "duration_seconds": round(duration, 2),
                    "judge_reasoning": reasoning,
                    "final_output": final_output,
                    "expected_output": expected_output
                })
                
                # save to file
                with open(RESULTS_FILE, "w", encoding="utf-8") as out_file:
                    json.dump(all_results, out_file, indent=4)

if __name__ == "__main__":
    run_benchmarks()
