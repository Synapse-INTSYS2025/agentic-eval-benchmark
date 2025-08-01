# uv run evaluate.py --output-file output/evaluation_results_with_metrics_llama32.csv --summary-file output/evaluation_summary_llama32.csv --report-file output/evaluation_final_report_llama32.txt
# uv run evaluate.py --output-file output/evaluation_results_with_metrics_mistral7b_instruct.csv --summary-file output/evaluation_summary_mistral7b_instruct.csv --report-file output/evaluation_final_report_mistral7b_instruct.txt
import json
import os
import argparse
from datetime import datetime
import requests
import pandas as pd
import nltk
from getpass import getpass

home_dir = os.path.expanduser("~")
NLTK_DATA_DIR = os.path.join(home_dir, 'nltk_data')
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)
nltk.data.path.append(NLTK_DATA_DIR)

cache_dir = os.path.join(home_dir, "my_hf_cache")
os.environ['HF_HOME'] = cache_dir
try:
    from huggingface_hub import login, HfFolder
    
    if HfFolder.get_token() is None:
        print("Hugging Face token not found in local cache.")
        hf_token = getpass("Please enter your Hugging Face Hub Read Token: ")
        
        if hf_token:
            print("Attempting to log in to Hugging Face Hub...")
            try:
                login(token=hf_token)
                print("✅ Login successful. Token has been cached for future runs.")
            except Exception as e:
                print(f"--- FATAL ERROR: Login failed. Please check your token. Error: {e} ---")
                exit()
        else:
            print("--- FATAL ERROR: No token provided. Exiting. ---")
            exit()
    else:
        print("✅ Hugging Face token found in cache. Proceeding with evaluation.")
        
except ImportError:
    print("WARNING: huggingface_hub library not found. The script may fail on gated models.")

from sacrebleu.metrics import TER
from comet import download_model, load_from_checkpoint

API_URL = "http://localhost:8088/natural_query"

def calculate_translation_metrics(generated_responses: dict, golden_row: pd.Series, comet_model) -> dict:
    """
    Calculates COMET, METEOR, and TER scores for each target language.
    """
    metrics = {}
    languages = ["english", "spanish", "italian", "luxembourgish", "portuguese", "french"]
    comet_data = []
    
    for lang in languages:
        golden_key = f"{lang}_summary"
        generated_translation = generated_responses.get(lang)
        golden_translation = golden_row.get(golden_key)
        
        if generated_translation and pd.notna(golden_translation):
            comet_data.append({
                "src": golden_row["english_summary"],
                "mt": generated_translation,
                "ref": golden_translation
            })
            
            reference_tokens = [nltk.word_tokenize(golden_translation)]
            candidate_tokens = nltk.word_tokenize(generated_translation)
            metrics[f"meteor_{lang}"] = nltk.translate.meteor_score.meteor_score(reference_tokens, candidate_tokens)
            
            ter_scorer = TER()
            ter_result = ter_scorer.corpus_score([generated_translation], [[golden_translation]])
            metrics[f"ter_{lang}"] = ter_result.score
        else:
            metrics[f"meteor_{lang}"] = None
            metrics[f"ter_{lang}"] = None

    if comet_data:
        model_output = comet_model.predict(comet_data, batch_size=8, gpus=0)
        scores = model_output.scores
        
        for i, data_item in enumerate(comet_data):
            lang_key = next((lang for lang in languages if f"{lang}_summary" in golden_row.index and golden_row.get(f"{lang}_summary") == data_item["ref"]), None)
            if lang_key:
                metrics[f"comet_{lang_key}"] = scores[i]

    return metrics

def send_request_and_collect_data(
    email_id: int, 
    consistency_run: int, 
    total_runs: int, 
    query_text: str, 
    expected_route: str,
    golden_summary_row: pd.Series, 
    comet_model
) -> dict:
    """
    Sends a single request to the API, collects all diagnostic data, calculates
    translation metrics, and returns a flat dictionary.
    """
    print(f"  Executing Consistency Run #{consistency_run}/{total_runs}...")
    
    request_payload = {"query": query_text}

    try:
        response = requests.post(
            API_URL, headers={"Content-Type": "application/json"},
            json=request_payload, timeout=180
        )
        response.raise_for_status()
        response_data = response.json()
        diagnostics = response_data.get("diagnostics", {})
        
        flat_result = {
            "email_id": email_id, "consistency_run": consistency_run, "expected_route": expected_route,
            "original_query": query_text, "timestamp_utc": datetime.utcnow().isoformat(),
            "http_status_code": response.status_code, "error_message": None
        }

        nlu_agent_data = diagnostics.get("nlu_agent", {})
        flat_result["nlu_extracted_route"] = nlu_agent_data.get("output", {}).get("parameters", {}).get("route_name")

        perf_agent_data = diagnostics.get("performance_agent", {})
        flat_result["performance_agent_output"] = json.dumps(perf_agent_data.get("output"))
        flat_result["performance_agent_error"] = perf_agent_data.get("error")

        work_agent_data = diagnostics.get("workload_agent", {})
        flat_result["workload_agent_output"] = json.dumps(work_agent_data.get("output"))
        flat_result["workload_agent_error"] = work_agent_data.get("error")
        
        narrative_agent_data = diagnostics.get("narrative_agent", {}).get("output", {})
        if isinstance(narrative_agent_data, dict):
            narrative_diagnostics = narrative_agent_data.get("diagnostics", {})
            flat_result["narrative_synthesis_llm_output"] = narrative_diagnostics.get("english_synthesis_llm", {}).get("output")
        else:
            flat_result["narrative_synthesis_llm_output"] = None

        final_response = response_data.get("response", {})
        
        languages_to_save = ["english", "spanish", "italian", "luxembourgish", "portuguese", "french"]
        for lang in languages_to_save:
            flat_result[f"final_{lang}_response"] = final_response.get(lang)
        
        if golden_summary_row is not None and not golden_summary_row.empty:
            translation_metrics = calculate_translation_metrics(final_response, golden_summary_row, comet_model)
            flat_result.update(translation_metrics)

        return flat_result

    except requests.exceptions.RequestException as e:
        print(f"  ERROR on Consistency Run #{consistency_run}: Request failed - {e}")
        return {"email_id": email_id, "consistency_run": consistency_run, "original_query": query_text, "error_message": str(e)}
    except Exception as e:
        print(f"  ERROR on Consistency Run #{consistency_run}: Failed to parse response - {e}")
        return {"email_id": email_id, "consistency_run": consistency_run, "original_query": query_text, "error_message": f"Response parsing error: {e}"}

def analyze_and_save_summary(raw_results_file: str, summary_file: str):
    """
    Reads the raw, run-by-run CSV data and computes aggregate statistics,
    saving them to a separate summary file.
    """
    print(f"\n--- Performing final analysis on results from '{raw_results_file}' ---")
    
    try:
        df = pd.read_csv(raw_results_file)
        if df.empty:
            print("Raw results file is empty. Skipping analysis.")
            return
    except FileNotFoundError:
        print(f"Raw results file not found at '{raw_results_file}'. Skipping analysis.")
        return

    df['nlu_success'] = df['nlu_extracted_route'] == df['expected_route']
    nlu_summary = df.groupby('expected_route')['nlu_success'].agg(
        nlu_success_rate='mean', 
        run_count='size'
    ).reset_index()

    metric_columns = [col for col in df.columns if col.startswith(('comet_', 'meteor_', 'ter_'))]
    if not metric_columns:
        print("No metric columns found. Skipping translation analysis.")
        summary_df = nlu_summary
    else:
        agg_funcs = ['mean', 'std', 'min', 'max']
        metrics_summary = df.groupby('expected_route')[metric_columns].agg(agg_funcs)
        metrics_summary.columns = ['_'.join(col).strip() for col in metrics_summary.columns.values]
        metrics_summary.reset_index(inplace=True)
        summary_df = pd.merge(nlu_summary, metrics_summary, on='expected_route', how='left')

    try:
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Route-by-route analysis summary saved to '{summary_file}'")
    except IOError as e:
        print(f"FATAL ERROR: Could not write analysis summary to file '{summary_file}'. Reason: {e}")

def create_final_report(summary_file: str, report_file: str):
    """
    Reads the aggregated summary CSV and computes the final "headline"
    statistics, saving them to a human-readable text file.
    """
    print(f"\n--- Creating final paper-ready report from '{summary_file}' ---")
    
    try:
        df = pd.read_csv(summary_file)
        if df.empty:
            print("Summary file is empty. Skipping final report.")
            return
    except FileNotFoundError:
        print(f"Summary file not found at '{summary_file}'. Skipping final report.")
        return

    report_lines = []
    languages_to_evaluate = ["english", "spanish", "french", "italian", "portuguese", "luxembourgish"]
    
    header = f"{'Metric':<20} | {'English':<12} | {'Spanish':<12} | {'French':<12} | {'Italian':<12} | {'Portuguese':<12} | {'Luxembourgish':<15}"
    separator = "-" * len(header)
    
    report_lines.append("="*len(header))
    report_lines.append("Overall System Output Quality Summary".center(len(header)))
    report_lines.append("="*len(header))
    report_lines.append(header)
    report_lines.append(separator)

    comet_means = [df[f'comet_{lang}_mean'].mean() for lang in languages_to_evaluate]
    report_lines.append(f"{'COMET (Mean)':<20} | " + " | ".join([f"{val:<12.3f}" for val in comet_means]))

    comet_stds = [df[f'comet_{lang}_std'].mean() for lang in languages_to_evaluate]
    report_lines.append(f"{'COMET (Std. Dev.)':<20} | " + " | ".join([f"{val:<12.3f}" for val in comet_stds]))
    
    meteor_means = [df[f'meteor_{lang}_mean'].mean() for lang in languages_to_evaluate]
    report_lines.append(f"{'METEOR (Mean)':<20} | " + " | ".join([f"{val:<12.3f}" for val in meteor_means]))

    ter_means = [df[f'ter_{lang}_mean'].mean() for lang in languages_to_evaluate]
    report_lines.append(f"{'TER (Mean)':<20} | " + " | ".join([f"{val:<12.2f}" for val in ter_means]))

    report_lines.append(separator)

    overall_nlu_success = df['nlu_success_rate'].mean()
    report_lines.append(f"\nOverall NLU Success Rate: {overall_nlu_success:.1%}")
    report_lines.append("\nNote: For COMET and METEOR, higher is better. For TER, lower is better.")
    report_lines.append("="*len(header))

    try:
        report_dir = os.path.dirname(report_file)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir)
        with open(report_file, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + "\n")
        print(f"✅ Final paper-ready report saved to '{report_file}'")
    except IOError as e:
        print(f"FATAL ERROR: Could not write final report to file '{report_file}'. Reason: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a resumable, multi-run consistency evaluation for each query.")
    parser.add_argument("-i", "--input-file", type=str, default="data/emails.json", help="Path to the input JSON file.")
    parser.add_argument("-n", "--num-runs", type=int, default=3, help="Number of times to run for EACH query.")
    parser.add_argument("-o", "--output-file", type=str, default="output/evaluation_results_with_metrics.csv", help="Path for the raw output CSV.")
    parser.add_argument("-g", "--golden-summaries", type=str, default="data/golden_summaries.csv", help="Path to the golden summaries CSV file.")
    parser.add_argument("-s", "--summary-file", type=str, default="output/evaluation_summary.csv", help="Path for the final aggregated analysis CSV.")
    parser.add_argument("-r", "--report-file", type=str, default="output/evaluation_final_report.txt", help="Path for the final paper-ready text report.")
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            email_queries = json.load(f)
        print(f"Successfully loaded {len(email_queries)} queries from '{args.input_file}'.")
        print(f"Each query will be run {args.num_runs} times.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not load or parse input file '{args.input_file}'. Error: {e}")
        return

    try:
        golden_df = pd.read_csv(args.golden_summaries)
        golden_df.set_index('route_name', inplace=True)
        print(f"Successfully loaded {len(golden_df)} golden summaries from '{args.golden_summaries}'.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Golden summaries file not found at '{args.golden_summaries}'. Metrics cannot be calculated.")
        return
    except KeyError:
        print(f"FATAL ERROR: The golden summaries file must contain a 'route_name' column.")
        return
        
    print("Setting up translation metrics models (this may take a moment on first run)...")
    try:
        nltk.download('punkt', quiet=True, download_dir=NLTK_DATA_DIR)
        nltk.download('wordnet', quiet=True, download_dir=NLTK_DATA_DIR)
        nltk.download('punkt_tab', quiet=True, download_dir=NLTK_DATA_DIR)
        
        comet_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
        comet_model = load_from_checkpoint(comet_model_path)
        print("Metrics models loaded successfully.")
    except Exception as e:
        print(f"--- FATAL ERROR: Could not download or load metrics models. Please ensure you are logged in. Error: {e} ---")
        return

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    completion_counts = pd.Series(dtype=int)
    if os.path.exists(args.output_file):
        print(f"Output file found at '{args.output_file}'. Attempting to resume.")
        try:
            existing_df = pd.read_csv(args.output_file)
            if not existing_df.empty:
                completion_counts = existing_df['email_id'].value_counts()
                print(f"Resuming based on {len(existing_df)} previously completed runs.")
            write_header = False
        except (pd.errors.EmptyDataError, FileNotFoundError):
            write_header = True
    else:
        write_header = True
    
    total_results_saved = 0
    for i, email_item in enumerate(email_queries, 1):
        num_done = completion_counts.get(i, 0)

        if num_done >= args.num_runs:
            print(f"--- Skipping Email #{i} (Route: {email_item.get('route')}) - Already completed {num_done}/{args.num_runs} runs. ---")
            continue

        print(f"\n--- Starting Evaluation for Email #{i} (Route: {email_item.get('route')}) ---")
        if num_done > 0:
            print(f"  Resuming from run #{num_done + 1}.")
        
        expected_route = email_item.get("route")
        golden_summary_row = golden_df.loc[expected_route] if expected_route in golden_df.index else None
        if golden_summary_row is None:
            print(f"  WARNING: Route '{expected_route}' not found in golden summaries. Metrics will not be calculated.")
            
        for run_num in range(num_done + 1, args.num_runs + 1):
            result = send_request_and_collect_data(
                email_id=i, consistency_run=run_num, total_runs=args.num_runs,
                query_text=email_item.get("query"), expected_route=expected_route,
                golden_summary_row=golden_summary_row, comet_model=comet_model
            )
            if result:
                df_single_result = pd.DataFrame([result])
                try:
                    df_single_result.to_csv(args.output_file, mode='a', header=write_header, index=False)
                    write_header = False
                    total_results_saved += 1
                except IOError as e:
                    print(f"\nFATAL ERROR: Could not write to file '{args.output_file}'. Reason: {e}")
                    return

    print(f"\nEvaluation complete. A total of {total_results_saved} new results were saved to '{args.output_file}'")
    
    if total_results_saved > 0 or os.path.exists(args.output_file):
         analyze_and_save_summary(args.output_file, args.summary_file)
         create_final_report(args.summary_file, args.report_file)

if __name__ == "__main__":
    main()