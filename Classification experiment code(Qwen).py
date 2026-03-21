import json
import os
import re
import io
import tokenize
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_java
import tree_sitter_go
import codebleu.utils
from codebleu import calc_codebleu
from dotenv import load_dotenv

load_dotenv()

# ================= Configuration =================
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = 
QWEN_MODEL_CANDIDATES = ["qwen-plus"]
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)
# Replace the placeholder strings below with your own paths before running.
INPUT_JSON = r"PUT_THE_FULL_PATH_TO_YOUR_INPUT_DATASET_JSON_HERE"
OUTPUT_CSV = r"PUT_THE_FULL_PATH_TO_THE_OUTPUT_CSV_FILE_HERE"
OUTPUT_JSON = r"PUT_THE_FULL_PATH_TO_THE_OUTPUT_JSON_FILE_HERE"
OUTPUT_LANG_AVG_CSV = r"PUT_THE_FULL_PATH_TO_THE_LANGUAGE_AVERAGE_CSV_FILE_HERE"
OUTPUT_API_TYPE_AVG_CSV = r"PUT_THE_FULL_PATH_TO_THE_API_TYPE_AVERAGE_CSV_FILE_HERE"
API_REF_JSON = r"PUT_THE_FULL_PATH_TO_YOUR_API_REFERENCE_JSON_HERE"
MAX_WORKERS = 8


# ================= 1. Tree-sitter Compatibility Patch =================
def patched_get_tree_sitter_language(lang: str):
    lang = lang.lower()
    try:
        if lang == "python": return tree_sitter.Language(tree_sitter_python.language())
        if lang in ["javascript", "js"]: return tree_sitter.Language(tree_sitter_javascript.language())
        if lang == "java": return tree_sitter.Language(tree_sitter_java.language())
        if lang == "go": return tree_sitter.Language(tree_sitter_go.language())
    except:
        return None
    return None


codebleu.utils.get_tree_sitter_language = patched_get_tree_sitter_language


# ================= 2. Helper Functions =================
def clean_code(text):
    pattern = r"```(?:python|javascript|js|java|go)?\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def strip_comments(code, lang):
    if not code:
        return code
    if lang == "python":
        code = re.sub(r"'''[\s\S]*?'''", "", code)
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        code = re.sub(r"#.*", "", code)
        return code
    # javascript/java/go: remove // and /* */
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)
    return code


def strip_key_info(code):
    if not code:
        return code
    # remove full URLs
    code = re.sub(r"https?://[^\s'\"<>]+", "", code)
    # remove host header values while keeping the key
    code = re.sub(r"(x-rapidapi-key['\"]?\s*[:=]\s*)['\"][^'\"]+['\"]", r"\1''", code, flags=re.IGNORECASE)
    code = re.sub(r"(X-RapidAPI-key['\"]?\s*[:=]\s*)['\"][^'\"]+['\"]", r"\1''", code)
    return code


def normalize_code_for_bleu(code, lang):
    code = strip_comments(code, lang)
    code = strip_key_info(code)
    return code


def normalize_lang(lang: str) -> str:
    lang = (lang or "").lower()
    if lang in ["javascript", "js"]:
        return "javascript"
    if lang in ["py", "python"]:
        return "python"
    return lang


def tokenize_code(code: str, lang: str):
    lang = normalize_lang(lang)
    if lang == "python":
        try:
            tokens = []
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.ENDMARKER,
                ):
                    continue
                if tok.string:
                    tokens.append(tok.string)
            return tokens
        except Exception:
            pass
    return re.findall(
        r"[A-Za-z_]\w*|\d+|===|!==|==|!=|<=|>=|=>|->|&&|\|\||[{}()\[\].,;:+\-*/%<>]",
        code,
    )


def _ngram_counts(tokens, n):
    counts = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def calc_bleu(ref_code, gen_code, lang):
    lang = normalize_lang(lang)
    ref = normalize_code_for_bleu(ref_code, lang)
    gen = normalize_code_for_bleu(gen_code, lang)
    ref_tokens = tokenize_code(ref, lang)
    gen_tokens = tokenize_code(gen, lang)
    if not gen_tokens:
        return 0.0
    precisions = []
    for n in range(1, 5):
        ref_counts = _ngram_counts(ref_tokens, n)
        gen_counts = _ngram_counts(gen_tokens, n)
        if not gen_counts:
            precisions.append(0.0)
            continue
        overlap = 0
        for ngram, cnt in gen_counts.items():
            overlap += min(cnt, ref_counts.get(ngram, 0))
        precisions.append(overlap / max(1, sum(gen_counts.values())))
    # brevity penalty
    ref_len = len(ref_tokens)
    gen_len = len(gen_tokens)
    if gen_len == 0:
        bp = 0.0
    elif gen_len > ref_len:
        bp = 1.0
    else:
        bp = pow(2.718281828, 1 - (ref_len / max(gen_len, 1)))
    # geometric mean with smoothing
    smooth = 1e-9
    score = bp
    for p in precisions:
        score *= max(p, smooth) ** 0.25
    return score


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def calc_es(ref_code, gen_code, lang):
    lang = normalize_lang(lang)
    ref = normalize_code_for_bleu(ref_code, lang)
    gen = normalize_code_for_bleu(gen_code, lang)
    ref = " ".join(tokenize_code(ref, lang))
    gen = " ".join(tokenize_code(gen, lang))
    if not ref and not gen:
        return 1.0
    if not ref or not gen:
        return 0.0
    dist = _levenshtein_distance(ref, gen)
    return max(0.0, 1.0 - dist / max(len(ref), len(gen)))


def extract_api_signals(code):
    if not code:
        return {
            "urls": set(),
            "hosts": set(),
            "methods": set()
        }
    urls = set(re.findall(r"https?://[^\s'\"<>]+", code))
    hosts = set()
    for url in urls:
        host_match = re.match(r"https?://([^/]+)", url)
        if host_match:
            hosts.add(host_match.group(1).lower())
    # common header host usage
    hosts.update(h.lower() for h in re.findall(r"[\"']X-RapidAPI-Host[\"']\s*[:=]\s*[\"']([^\"']+)[\"']", code, flags=re.IGNORECASE))
    # method signals
    methods = set(m.upper() for m in re.findall(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", code, flags=re.IGNORECASE))
    return {
        "urls": urls,
        "hosts": hosts,
        "methods": methods
    }


def calc_api_weighted_recall(ref_code, gen_code):
    ref_signals = extract_api_signals(ref_code)
    gen_signals = extract_api_signals(gen_code)

    def coverage(ref_set, gen_set):
        if not ref_set:
            return 1.0
        return len(ref_set & gen_set) / len(ref_set)

    url_score = coverage(ref_signals["urls"], gen_signals["urls"])
    host_score = coverage(ref_signals["hosts"], gen_signals["hosts"])
    method_score = coverage(ref_signals["methods"], gen_signals["methods"])

    # API correctness should dominate
    api_score = 0.6 * url_score + 0.3 * host_score + 0.1 * method_score
    return api_score


def load_api_reference(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        ref_map: Dict[str, List[Dict[str, Any]]] = {}
        for entry in data:
            for func_name in entry.get("function", []):
                key = str(func_name or "").strip().lower()
                if not key:
                    continue
                ref_map.setdefault(key, []).append(entry)
        return ref_map
    except Exception as e:
        print(f"Warning: failed to load API reference file: {e}")
        return {}


API_REF_MAP = load_api_reference(API_REF_JSON)


def _select_best_endpoint_metadata(func_name: str, endpoints: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not endpoints:
        return None
    normalized_name = re.sub(r"[_\-\s]+", " ", (func_name or "").lower()).strip()
    name_tokens = [t for t in re.split(r"[_\-\s]+", (func_name or "").lower()) if t]
    best_meta = None
    best_score = -1
    for meta in endpoints:
        desc = str(meta.get("description", "")).lower()
        score = 0
        if normalized_name and normalized_name in desc:
            score += 100
        for token in name_tokens:
            if token in desc:
                score += 1
        if meta.get("url"):
            score += 0.1
        if score > best_score:
            best_score = score
            best_meta = meta
    return best_meta


def get_api_metadata(item, lang):
    func_name = item.get('function_name')
    if not func_name:
        return None
    candidates = API_REF_MAP.get(str(func_name).strip().lower(), [])
    if not candidates:
        return None
    for entry in candidates:
        endpoints = entry.get("endpoints_metadata", []) or []
        selected = _select_best_endpoint_metadata(str(func_name), endpoints)
        if selected:
            metadata = {
                "method": selected.get("method"),
                "url": selected.get("url"),
                "headers": selected.get("headers"),
                "params": selected.get("params"),
                "payload": selected.get("payload"),
                "description": selected.get("description"),
                "api_name": entry.get("api_name"),
            }
            if metadata.get("url") or metadata.get("method"):
                return metadata
    for entry in candidates:
        endpoints = entry.get("endpoints_metadata", []) or []
        if endpoints:
            fallback = endpoints[0]
            metadata = {
                "method": fallback.get("method"),
                "url": fallback.get("url"),
                "headers": fallback.get("headers"),
                "params": fallback.get("params"),
                "payload": fallback.get("payload"),
                "description": fallback.get("description"),
                "api_name": entry.get("api_name"),
            }
            if metadata.get("url") or metadata.get("method"):
                return metadata
    return None

def build_api_metadata_hint(api_metadata: Dict[str, Any]) -> str:
    if not api_metadata:
        return ""
    return f"""

[API Metadata Reference]
1. api_name: {api_metadata.get('api_name')}
2. url: {api_metadata.get('url')}
3. method: {api_metadata.get('method')}
4. headers: {api_metadata.get('headers', {})}
5. params: {api_metadata.get('params')}
6. payload: {api_metadata.get('payload')}
7. description: {api_metadata.get('description')}"""


def build_prompt(item):
    """Build a language-specific prompt and prevent cross-language generation."""
    task_desc = item.get('input', '')
    func_name = item.get('function_name', 'solution')
    lang = item.get('language', 'python').lower()
    rely = ", ".join(item.get('rely', []))

    # Normalize the parameter field so mixed dict / str inputs are supported.
    param_list = []
    has_rapidapi = False
    raw_params = item.get('parameter', [])
    if not isinstance(raw_params, list):
        raw_params = [raw_params]
    for p in raw_params:
        p_name = None
        p_value = None
        if isinstance(p, dict):
            p_name = p.get('name')
            p_value = p.get('value')
        elif isinstance(p, str):
            p_name = p.strip()
        else:
            p_name = str(p)
        if not p_name:
            continue
        if "rapidapi" in str(p_name).lower():
            has_rapidapi = True
        if p_value is not None:
            # Preserve provided default values in the generated function signature.
            param_list.append(f"{p_name}={p_value}")
        else:
            param_list.append(str(p_name))
    params = ", ".join(param_list)

    api_metadata = get_api_metadata(item, lang)
    # Adjust the RapidAPI key rule depending on whether reference metadata is available.
    if api_metadata:
        rapidapi_key_rule = "\nGeneral rule: if the request headers include `X-RapidAPI-Key` (case-insensitive), generate it according to the referenced API metadata."
    else:
        rapidapi_key_rule = "\nGeneral security rule: if the request headers include `X-RapidAPI-Key` (case-insensitive), its value must be exactly `you-RapidAPI-key`. Never output a real secret."

    # Use strict language-specific instructions to avoid cross-language leakage.
    if lang in ["javascript", "js"]:
        lang_instruction = f"""
4. **You must use JavaScript (ES6+) syntax**. Do not use Python keywords such as `def`, `try:`, `except:`, or `None`.
5. **Function definition**: use either `export const {func_name} = ({params}) => {{ ... }}` or `function {func_name}({params}) {{ ... }}`.
6. **Async handling**: if the task involves API calls, use `async/await` or `Promise`.
7. **Context adaptation**: if the task mentions 'Ref' or 'doc', it may be a Firebase scenario, so prefer `doc(db, ...)`; if it mentions 'Btn' or 'tbody', it may be DOM-related.{rapidapi_key_rule}"""
    else:
        # Python-specific instructions
        rapidapi_hint = ""
        if has_rapidapi or "rapidapi" in task_desc.lower():
            rapidapi_hint = "\n4. Include the RapidAPI headers: 'X-RapidAPI-Key' and 'X-RapidAPI-Host'.\n5. Use the `requests` library and include a try-except block for `.json()`, assigning the result to `observation`."
        lang_instruction = f"{rapidapi_hint}\n6. Preserve valid Python indentation exactly.{rapidapi_key_rule}"

    api_metadata_hint = ""
    if api_metadata:
        api_metadata_hint = build_api_metadata_hint(api_metadata)

    prompt = f"""You are an expert {lang} software engineer. Generate code that follows the requirements below.

[Task]
{task_desc}

[Requirements]
1. Programming language: {lang.upper()}
2. Function or variable name: {func_name}
3. Parameter list: ({params})
4. Dependencies: {rely if rely else "native implementation only"}{lang_instruction}
{api_metadata_hint}

[Output Rules]
Return code only.
Do not include explanations.
Do not include Markdown code fences.
Output the raw code body directly."""
    return prompt, bool(api_metadata)


def is_generation_success(gen_code: str, lang: str) -> bool:
    gen_code = (gen_code or "").strip()
    if not gen_code:
        return False
    ts_lang = patched_get_tree_sitter_language((lang or "").lower())
    if not ts_lang:
        return True
    try:
        parser = tree_sitter.Parser(ts_lang)
        tree = parser.parse(bytes(gen_code, "utf8"))
        return not tree.root_node.has_error
    except Exception:
        return False

def call_qwen(prompt):
    try:
        last_error = None
        for model_name in QWEN_MODEL_CANDIDATES:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    return clean_code(content)
                return ""
            except Exception as e:
                last_error = e
                continue
        if last_error:
            raise last_error
        return ""
    except Exception as e:
        print(f"API call error: {e}")
        return ""


# ================= 3. Evaluation Engine =================
def evaluate_all(item, gen_code):
    ref_code = item.get('output', '')
    lang = item.get('language', 'python').lower()

    # --- 1. Syntax Pass ---
    ts_lang = patched_get_tree_sitter_language(lang)
    syntax_pass = False
    if ts_lang:
        try:
            parser = tree_sitter.Parser(ts_lang)
            tree = parser.parse(bytes(gen_code, "utf8"))
            syntax_pass = not tree.root_node.has_error
        except:
            syntax_pass = False

    # --- 2. CodeBLEU ---
    try:
        cb_lang = 'javascript' if lang in ['javascript', 'js'] else lang
        ref_code_for_bleu = normalize_code_for_bleu(ref_code, cb_lang)
        gen_code_for_bleu = normalize_code_for_bleu(gen_code, cb_lang)
        cb_res = calc_codebleu([ref_code_for_bleu], [gen_code_for_bleu], lang=cb_lang)
        cb_score = cb_res['codebleu']
    except:
        cb_score = 0.0

    # --- 2.1 BLEU ---
    try:
        bleu_score = calc_bleu(ref_code, gen_code, lang)
    except:
        bleu_score = 0.0

    # --- 2.2 ES (Exact Set / Exact Match) ---
    try:
        es_score = calc_es(ref_code, gen_code, lang)
    except:
        es_score = 0.0

    # --- 3. API Logic Recall ---
    # Extract important identifiers from the reference code.
    ref_tokens = set(re.findall(r'\b\w+\b', ref_code))
    gen_tokens = set(re.findall(r'\b\w+\b', gen_code))

    # Remove overly common tokens to make recall more meaningful.
    stop_words = {'const', 'let', 'var', 'function', 'def', 'return', 'import', 'from', 'if', 'try', 'except'}
    key_logic_tokens = [t for t in ref_tokens if t not in stop_words and len(t) > 2]

    if not key_logic_tokens:
        logic_recall = 1.0
    else:
        found = sum(1 for t in key_logic_tokens if t in gen_tokens)
        logic_recall = found / len(key_logic_tokens)

    api_recall = calc_api_weighted_recall(ref_code, gen_code)
    # API correctness gets higher weight
    combined_recall = 0.7 * api_recall + 0.3 * logic_recall

    if cb_score > 0 or combined_recall > 0:
        f1_score = (2 * cb_score * combined_recall) / (cb_score + combined_recall)
    else:
        f1_score = 0.0

    scale = 100.0
    return {
        "CodeBLEU": round(cb_score * scale, 4),
        "BLEU": round(bleu_score * scale, 4),
        "ES": round(es_score * scale, 4),
        "API_Recall": round(combined_recall * scale, 4),
        "F1": round(f1_score * scale, 4)
    }


# ================= 4. Main Loop =================
def format_control_record(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    return {
        "input": item.get("input", ""),
        "last_updated": item.get("last_updated"),
        "stars": item.get("stars"),
        "forks": item.get("forks"),
        "rely": item.get("rely", []),
        "function_name": item.get("function_name"),
        "parameter": item.get("parameter", []),
        "output": item.get("output", ""),
        "language": item.get("language", "python"),
        "api_type_category": item.get("api_type_category", "UNKNOWN"),
        "config_code": item.get("config_code", ""),
        "generated_code": item.get("generated_code", ""),
        "api_doc_referenced": bool(item.get("api_doc_referenced", False)),
        "api_doc_referenced_and_success": bool(item.get("api_doc_referenced_and_success", False)),
        "metrics": {
            "CodeBLEU": float(metrics.get("CodeBLEU", 0.0) or 0.0),
            "BLEU": float(metrics.get("BLEU", 0.0) or 0.0),
            "ES": float(metrics.get("ES", 0.0) or 0.0),
            "API_Recall": float(metrics.get("API_Recall", 0.0) or 0.0),
            "F1": float(metrics.get("F1", 0.0) or 0.0),
        },
    }


def has_valid_metrics(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    required = ("CodeBLEU", "BLEU", "ES", "API_Recall", "F1")
    return all(k in metrics for k in required)


def is_completed_sample(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    gen_code = str(item.get("generated_code", "") or "").strip()
    if not gen_code:
        return False
    return has_valid_metrics(item.get("metrics"))


def flatten_grouped_dataset(raw_data: Any) -> List[Dict[str, Any]]:
    """Support two input formats:
    1) legacy format: [sample, sample, ...]
    2) grouped format: [{"api_type_category": "...", "items": [sample, ...]}, ...]
    """
    if not isinstance(raw_data, list):
        return []
    if not raw_data:
        return []

    is_grouped = False
    first = raw_data[0]
    if isinstance(first, dict) and isinstance(first.get("items"), list):
        is_grouped = True

    if not is_grouped:
        return [x for x in raw_data if isinstance(x, dict)]

    flat: List[Dict[str, Any]] = []
    for group in raw_data:
        if not isinstance(group, dict):
            continue
        api_type = str(group.get("api_type_category", "UNKNOWN"))
        items = group.get("items", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["api_type_category"] = str(row.get("api_type_category", api_type))
            flat.append(row)
    return flat


def process_one_sample(i: int, item: Dict[str, Any]) -> Dict[str, Any]:
    prompt, used_api_doc_ref = build_prompt(item)
    gen_code = call_qwen(prompt)
    scores = evaluate_all(item, gen_code)
    generation_success = is_generation_success(gen_code, item.get('language', ''))
    return {
        "index": i,
        "generated_code": gen_code,
        "scores": scores,
        "used_api_doc_ref": used_api_doc_ref,
        "generation_success": generation_success,
    }


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = flatten_grouped_dataset(raw_data)
    if not dataset:
        print("Error: the input dataset is empty or not in the expected format.")
        return

    for out_path in (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_LANG_AVG_CSV, OUTPUT_API_TYPE_AVG_CSV):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    results_for_csv = []
    processed_ids = set()
    avg = {
        "CodeBLEU": 0.0,
        "BLEU": 0.0,
        "ES": 0.0,
        "API_Recall": 0.0,
        "F1": 0.0,
    }
    lang_avg = {
        "python": {k: 0.0 for k in avg},
        "javascript": {k: 0.0 for k in avg},
    }
    lang_count = {"python": 0, "javascript": 0}
    api_type_avg: Dict[str, Dict[str, float]] = defaultdict(lambda: {k: 0.0 for k in avg})
    api_type_count: Dict[str, int] = defaultdict(int)
    ref_used_count = 0
    ref_used_and_success_count = 0
    skipped_count = 0

    # Resume from checkpoint if an intermediate output file already exists.
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            if isinstance(saved_data, list):
                # Keep api_type_category from current input dataset if missing in checkpoint records.
                merged_dataset = list(dataset)
                for idx, saved_item in enumerate(saved_data):
                    if idx >= len(merged_dataset):
                        break
                    if isinstance(saved_item, dict):
                        merged = dict(merged_dataset[idx]) if isinstance(merged_dataset[idx], dict) else {}
                        merged.update(saved_item)
                        if not str(merged.get("api_type_category", "")).strip():
                            merged["api_type_category"] = (
                                merged_dataset[idx].get("api_type_category", "UNKNOWN")
                                if isinstance(merged_dataset[idx], dict)
                                else "UNKNOWN"
                            )
                        merged_dataset[idx] = merged
                dataset = merged_dataset
                for idx, item in enumerate(dataset):
                    if not is_completed_sample(item):
                        continue
                    metrics = item.get("metrics")
                    rec = {
                        "ID": idx + 1,
                        "Function": item.get('function_name'),
                        "Language": item.get('language'),
                        "API_Type": item.get("api_type_category", "UNKNOWN"),
                        "API_Doc_Referenced": bool(item.get('api_doc_referenced', False)),
                        "API_Doc_Referenced_And_Success": bool(item.get('api_doc_referenced_and_success', False)),
                        "CodeBLEU": metrics.get("CodeBLEU", 0.0),
                        "BLEU": metrics.get("BLEU", 0.0),
                        "ES": metrics.get("ES", 0.0),
                        "API_Recall": metrics.get("API_Recall", 0.0),
                        "F1": metrics.get("F1", 0.0),
                    }
                    results_for_csv.append(rec)
                    processed_ids.add(idx + 1)
                # Restore aggregate statistics from completed records.
                for rec in results_for_csv:
                    for k in avg:
                        avg[k] += float(rec.get(k, 0.0) or 0.0)
                    lang = normalize_lang(rec.get("Language"))
                    if lang in ("python", "javascript"):
                        for k in avg:
                            lang_avg[lang][k] += float(rec.get(k, 0.0) or 0.0)
                        lang_count[lang] += 1
                    api_type = str(rec.get("API_Type") or "UNKNOWN")
                    for k in avg:
                        api_type_avg[api_type][k] += float(rec.get(k, 0.0) or 0.0)
                    api_type_count[api_type] += 1
                    if rec.get("API_Doc_Referenced"):
                        ref_used_count += 1
                        if rec.get("API_Doc_Referenced_And_Success"):
                            ref_used_and_success_count += 1
                if processed_ids:
                    print(f"Resume detected: restored {len(processed_ids)} completed samples and will continue the remaining ones.")
        except Exception as e:
            print(f"Warning: failed to read checkpoint data, restarting from scratch: {e}")

    print(f"Starting multilingual API evaluation ({len(dataset)} samples)...")

    def calc_avg_snapshot() -> Dict[str, float]:
        processed = len(results_for_csv)
        if processed == 0:
            return {k: 0.0 for k in avg}
        return {k: round(avg[k] / processed, 4) for k in avg}

    def save_progress(reason: str) -> None:
        avg_snapshot = calc_avg_snapshot()
        pd.DataFrame(results_for_csv).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        lang_rows = []
        for lang in ("python", "javascript"):
            count = lang_count[lang] or 1
            row = {"Language": lang, "Count": lang_count[lang]}
            for k in avg_snapshot:
                row[k] = round(lang_avg[lang][k] / count, 4)
            lang_rows.append(row)
        pd.DataFrame(lang_rows).to_csv(OUTPUT_LANG_AVG_CSV, index=False, encoding='utf-8-sig')
        api_type_rows = []
        for api_type, count in sorted(api_type_count.items(), key=lambda x: x[1], reverse=True):
            row = {"API_Type": api_type, "Count": count}
            for k in avg_snapshot:
                row[k] = round(api_type_avg[api_type][k] / max(1, count), 4)
            api_type_rows.append(row)
        pd.DataFrame(api_type_rows).to_csv(OUTPUT_API_TYPE_AVG_CSV, index=False, encoding='utf-8-sig')
        formatted_dataset = [
            format_control_record(item) if isinstance(item, dict) else item
            for item in dataset
        ]
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(formatted_dataset, f, ensure_ascii=False, indent=4)
        print(f"Progress saved ({reason}): {len(results_for_csv)}/{len(dataset)}")

    try:
        pending = [(i, item) for i, item in enumerate(dataset) if (i + 1) not in processed_ids]
        print(f"Parallel mode started | workers={MAX_WORKERS} | pending={len(pending)}")
        newly_processed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_one_sample, i, item): (i, item)
                for i, item in pending
            }
            for future in concurrent.futures.as_completed(futures):
                i, item = futures[future]
                try:
                    result = future.result()
                    scores = result["scores"]
                    used_api_doc_ref = bool(result["used_api_doc_ref"])
                    generation_success = bool(result["generation_success"])
                    gen_code = str(result["generated_code"] or "")

                    if used_api_doc_ref:
                        ref_used_count += 1
                        if generation_success:
                            ref_used_and_success_count += 1

                    lang_label = str(item.get('language') or "unknown").upper()
                    func_label = str(item.get('function_name') or "unknown")
                    print(
                        f"[{i + 1:02d}/{len(dataset)}] {lang_label} | API: {func_label:<25} | CodeBLEU: {scores['CodeBLEU']:.4f}")

                    item['generated_code'] = gen_code
                    item['api_doc_referenced'] = used_api_doc_ref
                    item['api_doc_referenced_and_success'] = used_api_doc_ref and generation_success
                    item['metrics'] = scores
                    item.pop('error', None)
                    if not is_completed_sample(item):
                        raise ValueError("Generated code is empty or metric fields are invalid. The sample will remain for a later retry.")
                    results_for_csv.append({
                        "ID": i + 1,
                        "Function": item.get('function_name'),
                        "Language": item.get('language'),
                        "API_Type": item.get("api_type_category", "UNKNOWN"),
                        "API_Doc_Referenced": used_api_doc_ref,
                        "API_Doc_Referenced_And_Success": used_api_doc_ref and generation_success,
                        "CodeBLEU": scores['CodeBLEU'],
                        "BLEU": scores['BLEU'],
                        "ES": scores['ES'],
                        "API_Recall": scores['API_Recall'],
                        "F1": scores['F1']
                    })
                    processed_ids.add(i + 1)
                    for k in avg:
                        avg[k] += scores[k]
                    lang = (item.get('language') or "").lower()
                    if lang in ("python", "javascript", "js"):
                        if lang == "js":
                            lang = "javascript"
                        for k in avg:
                            lang_avg[lang][k] += scores[k]
                        lang_count[lang] += 1
                    api_type = str(item.get("api_type_category", "UNKNOWN"))
                    for k in avg:
                        api_type_avg[api_type][k] += scores[k]
                    api_type_count[api_type] += 1
                except Exception as item_err:
                    skipped_count += 1
                    item['error'] = str(item_err)
                    print(f"Warning: [{i + 1:02d}/{len(dataset)}] sample processing failed and was skipped: {item_err}")
                finally:
                    newly_processed += 1
                    if newly_processed % 100 == 0:
                        save_progress(f"checkpoint@{newly_processed}")
    except KeyboardInterrupt:
        print("\nInterrupted manually. Saving completed progress...")
        save_progress("interrupted")
        print("Progress before interruption has been saved.")
        return
    except Exception as e:
        print(f"\nRuntime error: {e}")
        print("Saving completed progress...")
        save_progress("exception")
        raise

    final_avg = calc_avg_snapshot()
    save_progress("final")
    print(f"\nEvaluation complete. Results saved to: {OUTPUT_CSV}")
    print(
        "Averages | "
        f"CodeBLEU: {final_avg['CodeBLEU']} | BLEU: {final_avg['BLEU']} | ES: {final_avg['ES']} | "
        f"API_Recall: {final_avg['API_Recall']} | F1: {final_avg['F1']}"
    )
    print(
        "API reference stats | "
        f"Reference hits: {ref_used_count} | "
        f"Reference hits with successful generation: {ref_used_and_success_count} | "
        f"Success rate: {round(ref_used_and_success_count / ref_used_count * 100, 2) if ref_used_count else 0.0}%"
    )
    print(f"Skipped samples due to exceptions: {skipped_count}")
    print(f"Language average metrics saved to: {OUTPUT_LANG_AVG_CSV}")
    print(f"API type average metrics saved to: {OUTPUT_API_TYPE_AVG_CSV}")


if __name__ == "__main__":
    main()
