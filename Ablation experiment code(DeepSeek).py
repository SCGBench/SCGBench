import json
import os
import re
import io
import tokenize
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_MODEL = "deepseek-chat"
INPUT_JSON = r"PUT_THE_FULL_PATH_TO_YOUR_INPUT_DATASET_JSON_HERE"
OUTPUT_CSV = r"PUT_THE_FULL_PATH_TO_THE_OUTPUT_CSV_FILE_HERE"
OUTPUT_JSON = r"PUT_THE_FULL_PATH_TO_THE_OUTPUT_JSON_FILE_HERE"
OUTPUT_LANG_AVG_CSV = r"PUT_THE_FULL_PATH_TO_THE_LANGUAGE_AVERAGE_CSV_FILE_HERE"
OUTPUT_ABLATION_CSV = r"PUT_THE_FULL_PATH_TO_THE_ABLATION_SUMMARY_CSV_FILE_HERE"
API_REF_JSON = r"PUT_THE_FULL_PATH_TO_YOUR_API_REFERENCE_JSON_HERE"


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


# ================= 2. Utility Functions =================
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
        print(f"Warning: Failed to load API reference: {e}")
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

API Metadata Reference:
1. api_name: {api_metadata.get('api_name')}
2. url: {api_metadata.get('url')}
3. method: {api_metadata.get('method')}
4. headers: {api_metadata.get('headers', {})}
5. params: {api_metadata.get('params')}
6. payload: {api_metadata.get('payload')}
7. description: {api_metadata.get('description')}"""


def build_prompt_ablation(item, setting_name: str, use_d: bool, use_s: bool, use_p: bool, use_dep: bool):
    """Ablation experiment prompt: D+S / D+S+P / D+S+P+Dep (P=Parameters)"""
    task_desc = item.get('input', '')
    func_name = item.get('function_name', 'solution')
    lang = normalize_lang(item.get('language', 'python'))

    param_list = []
    raw_params = item.get("parameter", [])
    if not isinstance(raw_params, list):
        raw_params = [raw_params]
    for p in raw_params:
        p_name = None
        p_value = None
        if isinstance(p, dict):
            p_name = p.get("name")
            p_value = p.get("value")
        elif isinstance(p, str):
            p_name = p.strip()
        else:
            p_name = str(p)
        if p_name is None or str(p_name).strip() == "":
            continue
        if p_value is not None:
            param_list.append(f"{p_name}={p_value}")
        else:
            param_list.append(str(p_name))
    params = ", ".join(param_list)
    config_code = (item.get("config_code") or "").strip()

    api_metadata = get_api_metadata(item, lang)
    api_metadata_hint = ""
    if api_metadata:
        api_metadata_hint = build_api_metadata_hint(api_metadata)

    language_label = "JavaScript (ES6+)" if lang == "javascript" else "Python"
    sections = [f"[Base]\nLanguage: {language_label}. Use {language_label} syntax only."]
    if use_d:
        sections.append(f"[D: Description]\n{task_desc}")
    if use_s:
        sections.append(f"[S: Signature]\nFunction name: {func_name}")
    if use_p:
        sections.append(f"[P: Parameters]\n{params if params else '(no parameters)'}")
    if use_dep:
        sections.append(f"[Dep: Config Code]\n{config_code if config_code else '(no config_code provided)'}")
    if api_metadata_hint:
        sections.append(f"[API Docs]\n{api_metadata_hint.strip()}")
    if use_dep:
        sections.append(
            "[Dep Instruction]\nWhen generating request code, place Config Code before the function and follow it inside the implementation."
        )
    prompt_body = "\n\n".join(sections).strip()
    prompt = f"""You are a senior {lang} software engineer. Follow all sections carefully.
Setting: {setting_name}

{prompt_body}

Output code only. No explanations. No Markdown code fences."""
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

def call_deepseek(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY or os.getenv('DEEPSEEK_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        response = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            return clean_code(content)
        return ""
    except Exception as e:
        print(f"API call error: {e}")
        return ""


# ================= 3. Evaluation Engine =================
def evaluate_all(item, gen_code):
    ref_code = item.get('output', '')
    lang = item.get('language', 'python').lower()

    # --- 2. CodeBLEU ---
    try:
        cb_lang = 'javascript' if lang in ['javascript', 'js'] else lang
        ref_code_for_bleu = normalize_code_for_bleu(ref_code, cb_lang)
        gen_code_for_bleu = normalize_code_for_bleu(gen_code, cb_lang)
        cb_res = calc_codebleu([ref_code_for_bleu], [gen_code_for_bleu], lang=cb_lang)
        cb_score = cb_res['codebleu']
    except:
        cb_score = 0.0

    # --- 2.2 ES ---
    try:
        es_score = calc_es(ref_code, gen_code, lang)
    except:
        es_score = 0.0

    scale = 100.0
    return {
        "CodeBLEU": round(cb_score * scale, 4),
        "ES": round(es_score * scale, 4),
    }


# ================= 4. Main Loop =================
def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for out_path in (OUTPUT_CSV, OUTPUT_JSON, OUTPUT_LANG_AVG_CSV, OUTPUT_ABLATION_CSV):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    ablation_settings = [
        ("D+S", dict(use_d=True, use_s=True, use_p=False, use_dep=False)),
        ("D+S+P", dict(use_d=True, use_s=True, use_p=True, use_dep=False)),
        ("D+S+P+Dep", dict(use_d=True, use_s=True, use_p=True, use_dep=True)),
    ]

    rows = []
    sample_rows = []
    state = {
        "completed_settings": [],
        "sample_records_count": 0,
        "summary_rows_count": 0,
    }
    
    completed_settings = set()
    processed_sample_keys = set()

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                loaded_state = saved.get("state", {}) or {}
                loaded_samples = saved.get("sample_records", []) or []
                loaded_rows = saved.get("summary_rows", []) or []
                if isinstance(loaded_samples, list):
                    sample_rows = loaded_samples
                if isinstance(loaded_rows, list):
                    rows = loaded_rows
                if isinstance(loaded_state, dict):
                    state.update(loaded_state)
                completed_settings = set(state.get("completed_settings", []) or [])
                for rec in sample_rows:
                    if not isinstance(rec, dict):
                        continue
                    s = rec.get("Setting")
                    idx = rec.get("Index")
                    if s is None or idx is None:
                        continue
                    try:
                        processed_sample_keys.add((str(s), int(idx)))
                    except Exception:
                        continue
                if sample_rows or rows or completed_settings:
                    print(
                        f"Checkpoint detected: completed_settings={len(completed_settings)} | "
                        f"sample={len(sample_rows)} | summary={len(rows)}"
                    )
        except Exception as e:
            print(f"Warning: Failed to load checkpoint, starting from scratch: {e}")

    print(f"Starting ablation experiment ({len(dataset)} samples)...")

    def save_progress(reason: str) -> None:
        pd.DataFrame(sample_rows).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        lang_avg_rows = [r for r in rows if r.get("Language") in ("python", "javascript", "average")]
        pd.DataFrame(lang_avg_rows).to_csv(OUTPUT_LANG_AVG_CSV, index=False, encoding='utf-8-sig')
        state["sample_records_count"] = len(sample_rows)
        state["summary_rows_count"] = len(rows)
        state["reason"] = reason
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "state": state,
                    "sample_records": sample_rows,
                    "summary_rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        pd.DataFrame(rows).to_csv(OUTPUT_ABLATION_CSV, index=False, encoding='utf-8-sig')
        print(
            f"Progress saved ({reason}): "
            f"sample={len(sample_rows)} | summary={len(rows)}"
        )

    try:
        for setting_name, flags in ablation_settings:
            if setting_name in completed_settings:
                print(f"\n==== Skipping completed setting: {setting_name} ====")
                continue
            print(f"\n==== Running setting: {setting_name} ====")
            lang_sum = {
                "python": {"CodeBLEU": 0.0, "ES": 0.0},
                "javascript": {"CodeBLEU": 0.0, "ES": 0.0},
            }
            lang_success = {"python": 0, "javascript": 0}
            
            for rec in sample_rows:
                if not isinstance(rec, dict):
                    continue
                if rec.get("Setting") != setting_name:
                    continue
                if rec.get("Status") != "success":
                    continue
                lang = normalize_lang(rec.get("Language"))
                if lang not in ("python", "javascript"):
                    continue
                try:
                    lang_sum[lang]["CodeBLEU"] += float(rec.get("CodeBLEU") or 0.0)
                    lang_sum[lang]["ES"] += float(rec.get("ES") or 0.0)
                    lang_success[lang] += 1
                except Exception:
                    continue

            for i, item in enumerate(dataset):
                if (setting_name, i + 1) in processed_sample_keys:
                    continue
                try:
                    lang = normalize_lang(item.get('language'))
                    if lang not in ("python", "javascript"):
                        print(
                            f"[{setting_name}] [{i + 1}/{len(dataset)}] "
                            f"lang={lang or 'unknown'} | status=skipped(unsupported-language)"
                        )
                        continue
                    prompt, _ = build_prompt_ablation(item, setting_name, **flags)
                    gen_code = call_deepseek(prompt)
                    if not is_generation_success(gen_code, lang):
                        sample_rows.append(
                            {
                                "Setting": setting_name,
                                "Index": i + 1,
                                "Function": item.get("function_name"),
                                "Language": lang,
                                "Status": "failed-generation",
                                "CodeBLEU": None,
                                "ES": None,
                            }
                        )
                        processed_sample_keys.add((setting_name, i + 1))
                        print(
                            f"[{setting_name}] [{i + 1}/{len(dataset)}] "
                            f"lang={lang} | status=failed-generation"
                        )
                        continue

                    scores = evaluate_all(item, gen_code)
                    for metric in ("CodeBLEU", "ES"):
                        lang_sum[lang][metric] += scores[metric]
                    lang_success[lang] += 1
                    sample_rows.append(
                        {
                            "Setting": setting_name,
                            "Index": i + 1,
                            "Function": item.get("function_name"),
                            "Language": lang,
                            "Status": "success",
                            "CodeBLEU": scores["CodeBLEU"],
                            "ES": scores["ES"],
                        }
                    )
                    processed_sample_keys.add((setting_name, i + 1))
                    print(
                        f"[{setting_name}] [{i + 1}/{len(dataset)}] "
                        f"lang={lang} | status=success | CodeBLEU={scores['CodeBLEU']:.2f} | ES={scores['ES']:.2f}"
                    )
                except Exception as item_err:
                    sample_rows.append(
                        {
                            "Setting": setting_name,
                            "Index": i + 1,
                            "Function": item.get("function_name"),
                            "Language": normalize_lang(item.get("language")) or "unknown",
                            "Status": "error-skipped",
                            "CodeBLEU": None,
                            "ES": None,
                            "Error": str(item_err),
                        }
                    )
                    processed_sample_keys.add((setting_name, i + 1))
                    print(
                        f"[{setting_name}] [{i + 1}/{len(dataset)}] "
                        f"status=error-skipped | error={item_err}"
                    )
                    continue

            
            rows = [r for r in rows if r.get("Setting") != setting_name]
            setting_rows = {}
            for lang in ("python", "javascript"):
                count = lang_success[lang]
                row = {
                    "Setting": setting_name,
                    "Language": lang,
                    "SuccessCount": count,
                    "CodeBLEU": round(lang_sum[lang]["CodeBLEU"] / count, 4) if count else 0.0,
                    "ES": round(lang_sum[lang]["ES"] / count, 4) if count else 0.0,
                }
                rows.append(row)
                setting_rows[lang] = row

            valid_rows = [r for r in (setting_rows["python"], setting_rows["javascript"]) if r["SuccessCount"] > 0]
            avg_row = {
                "Setting": setting_name,
                "Language": "average",
                "SuccessCount": sum(r["SuccessCount"] for r in valid_rows),
                "CodeBLEU": round(sum(r["CodeBLEU"] for r in valid_rows) / len(valid_rows), 4) if valid_rows else 0.0,
                "ES": round(sum(r["ES"] for r in valid_rows) / len(valid_rows), 4) if valid_rows else 0.0,
            }
            rows.append(avg_row)
            print(
                f"[{setting_name}] done | "
                f"python(success)={lang_success['python']} | javascript(success)={lang_success['javascript']}"
            )
            completed_settings.add(setting_name)
            state["completed_settings"] = sorted(completed_settings)
            
            save_progress(f"after-{setting_name}")
    except KeyboardInterrupt:
        print("\nInterrupt detected, saving completed results...")
        save_progress("interrupted")
        print("Progress saved before interruption.")
        return
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Saving completed results...")
        save_progress("exception")
        raise

    save_progress("final")
    print(f"\nAblation experiment completed! Results saved to: {OUTPUT_ABLATION_CSV}")


if __name__ == "__main__":
    main()
