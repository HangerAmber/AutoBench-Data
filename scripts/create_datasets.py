import os
import csv
import json
import sys
import random

try:
    import openai
except ImportError:
    openai = None
anthropic = None
AutoTokenizer = None
AutoModelForCausalLM = None

# Base class for LLM agent with multi-provider support
class BaseAgent:
    def __init__(self, model_name: str):
        """
        Initialize the agent with a given model name, determining the provider and setting up API access.
        """
        self.model_name = model_name
        # Determine provider by keywords in model_name
        name_lower = model_name.lower()
        if "gpt" in name_lower or name_lower.startswith("openai"):
            self.provider = "openai"
        elif "claude" in name_lower or "anthropic" in name_lower:
            self.provider = "anthropic"
        elif "qwen" in name_lower:
            self.provider = "qwen"
        else:
            self.provider = "openai"  # default to OpenAI if unrecognized
        # Set up API keys or model clients for the provider
        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            if openai is not None:
                openai.api_key = self.api_key
        elif self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
            try:
                import anthropic as anth
            except ImportError:
                raise ImportError("Anthropic API client not installed. Please install 'anthropic' package.")
            global anthropic
            anthropic = anth
            self.client = anthropic.Client(api_key=self.api_key)
        elif self.provider == "qwen":
            # Use Hugging Face transformers for Qwen
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # Load Qwen model and tokenizer (using the model_name or default to Qwen-7B-Chat)
            model_path = model_name if "qwen" in model_name.lower() else "Qwen/Qwen-7B-Chat"
            if not hasattr(BaseAgent, "_qwen_tokenizer"):
                BaseAgent._qwen_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                BaseAgent._qwen_model = AutoModelForCausalLM.from_pretrained(model_path)
            self.qwen_tokenizer = BaseAgent._qwen_tokenizer
            self.qwen_model = BaseAgent._qwen_model
        else:
            raise ValueError(f"Unsupported LLM provider for model: {model_name}")
        # Default generation parameters
        self.temperature = 0.7
        self.max_tokens = 512

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Invoke the underlying LLM with an optional system prompt and a user prompt.
        Returns the assistant's response text.
        """
        if self.provider == "openai":
            if openai is None:
                raise RuntimeError("OpenAI library not available.")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            result_text = response['choices'][0]['message']['content']
            return result_text.strip()
        elif self.provider == "anthropic":
            combined_prompt = user_prompt
            if system_prompt:
                combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt}"
            prompt_text = anthropic.HUMAN_PROMPT + " " + combined_prompt + anthropic.AI_PROMPT
            response = self.client.completion(
                prompt=prompt_text,
                model=self.model_name,
                max_tokens_to_sample=self.max_tokens,
                temperature=self.temperature
            )
            result_text = response.get('completion', '')
            return result_text.strip()
        elif self.provider == "qwen":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            # Use Qwen's chat template for formatting
            text_input = self.qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.qwen_tokenizer([text_input], return_tensors="pt")
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature
            )
            # Decode only the newly generated tokens
            output_ids = outputs[0][inputs['input_ids'].shape[1]:]
            result_text = self.qwen_tokenizer.decode(output_ids, skip_special_tokens=True)
            return result_text.strip()
        else:
            raise ValueError(f"Provider {self.provider} not supported.")

# Define agent classes for SPA, AGA, FIA, MEA
class SemanticParsingAgent(BaseAgent):
    def __init__(self, model_name: str, protocol_path: str = "protocol.json"):
        super().__init__(model_name)
        # Load protocol constraints from JSON file
        with open(protocol_path, "r", encoding="utf-8") as f:
            self.protocol = json.load(f)
        # Include {protocol} placeholder in the template
        self.system_prompt_template = (
            "Given the command: {c} and the protocol constraints: {protocol}, "
            "extract the intent, entities, and parameters in the format (I, E, P)."
        )

    def parse_instruction(self, instruction: str) -> dict:
        # Fill in both {c} and {protocol}
        system_prompt = self.system_prompt_template.format(
            c=instruction,
            protocol=json.dumps(self.protocol)
        )
        user_prompt = ""  # no additional user prompt
        result = self.call_llm(system_prompt, user_prompt)

        # Attempt to parse JSON from the model’s response
        try:
            jstart = result.index('{')
            jstr = result[jstart: result.rfind('}') + 1]
            parsed = json.loads(jstr)
            return parsed if isinstance(parsed, dict) else {"target_call": parsed}
        except Exception:
            # Fallback to returning raw text
            return {"target_call": result.strip()}


class AdversarialGenerationAgent(BaseAgent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # System prompt for AGA from paper appendix
        self.system_prompt_template = (
            "Given the command: {c}, refer to the slot information provided in the protocol constraints, "
            "generalize the seed query, and generate an adversarial variant by introducing extreme ambiguity "
            "while keeping it plausible."
        )

    def generate_ambiguous(self, clear_instruction: str, target_call: dict) -> dict:
        # Fill in the {c} placeholder with the clear instruction
        system_prompt = self.system_prompt_template.format(c=clear_instruction)
        # Include target_call as part of the user prompt for context
        user_prompt = f"Target call: {json.dumps(target_call)}"
        result = self.call_llm(system_prompt, user_prompt)

        try:
            return json.loads(result)
        except Exception:
            # Fallback parsing if JSON is not directly returned
            ambiguous_instruction = ""
            ambiguity_type = ""
            for line in result.splitlines():
                if "ambiguous_instruction" in line:
                    ambiguous_instruction = line.split(":", 1)[1].strip().strip('"')
                if "ambiguity_type" in line:
                    ambiguity_type = line.split(":", 1)[1].strip().strip('"')
            return {
                "ambiguous_instruction": ambiguous_instruction or result.strip(),
                "ambiguity_type": ambiguity_type or "unspecified"
            }


class FuzzInjectionAgent(BaseAgent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # System prompt for FIA from paper appendix
        self.system_prompt_template = (
            "Given the command: {c}, introduce ambiguity by {f} with intensity {ε}, "
            "where {f} is one of {omit parameter, subjective expression, ...}."
        )

    def inject_fuzz_and_clarify(self, ambiguous_instruction: str, ambiguity_type: str) -> dict:
        # Randomly choose a fuzz strategy and intensity
        f_options = ["omit parameter", "subjective expression", "syntactic distortion"]
        f_choice = random.choice(f_options)
        ε = round(random.uniform(0.3, 0.7), 2)

        # Fill placeholders {c}, {f}, and {ε}
        system_prompt = self.system_prompt_template.format(
            c=ambiguous_instruction,
            f=f_choice,
            ε=ε
        )
        user_prompt = ""  # No extra user prompt needed
        result = self.call_llm(system_prompt, user_prompt)

        try:
            return json.loads(result)
        except Exception:
            # Fallback parsing if JSON is not directly returned
            fuzzy_instruction = ""
            assistant_question = ""
            for line in result.splitlines():
                if "fuzzy_instruction" in line:
                    fuzzy_instruction = line.split(":", 1)[1].strip().strip('"')
                if "assistant_question" in line:
                    assistant_question = line.split(":", 1)[1].strip().strip('"')
            return {
                "fuzzy_instruction": fuzzy_instruction or ambiguous_instruction,
                "assistant_question": assistant_question
            }


class MultiTurnEvolutionAgent(BaseAgent):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # System prompt for MEA from paper appendix
        self.system_prompt_template = (
            "Given the dialogue history: {history}, generate the next system response r_t and user command c_{t+1}."
        )

    def expand_to_dialogue(self, initial_user: str, assistant_question: str, target_call: dict) -> list:
        # Build a string representation of the dialogue history
        history = [(initial_user, assistant_question)]
        hist_str = ", ".join(f"({repr(u)}, {repr(a)})" for u, a in history)

        # Fill the {history} placeholder
        system_prompt = self.system_prompt_template.format(history=hist_str)
        user_prompt = f"Target call: {json.dumps(target_call)}"
        result = self.call_llm(system_prompt, user_prompt)

        # Try parsing the response as a JSON list
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Fallback to line-by-line parsing
        dialogue = []
        for line in result.splitlines():
            line = line.strip()
            if line.lower().startswith("user:"):
                dialogue.append({"role": "user", "content": line.split(":", 1)[1].strip()})
            elif line.lower().startswith("assistant:"):
                dialogue.append({"role": "assistant", "content": line.split(":", 1)[1].strip()})

        # Ensure initial turns are present
        if not dialogue or dialogue[0]['role'] != 'user':
            dialogue.insert(0, {"role": "assistant", "content": assistant_question})
            dialogue.insert(0, {"role": "user", "content": initial_user})

        return dialogue

def generate_data(input_csv: str, model_name: str):
    # Initialize agents
    spa = SemanticParsingAgent(model_name)
    aga = AdversarialGenerationAgent(model_name)
    fia = FuzzInjectionAgent(model_name)
    mea = MultiTurnEvolutionAgent(model_name)
    tier1_data = []
    tier2_data = []
    tier3_data = []
    # Read seed instructions
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        seeds = []
        for row in reader:
            if not row: 
                continue
            instr = row[0].strip()
            # Skip header or invalid lines
            if instr.lower().startswith("instruction") or instr.lower().startswith("query"):
                continue
            seeds.append(instr)
    # Process each seed
    for idx, seed in enumerate(seeds, start=1):
        # Tier 1
        parse_result = spa.parse_instruction(seed)
        target_call = parse_result.get("target_call", parse_result)
        if target_call is None:
            target_call = {}
        ambig = aga.generate_ambiguous(seed, target_call)
        ambiguous_instruction = ambig.get("ambiguous_instruction", "").strip()
        ambiguity_type = ambig.get("ambiguity_type", "unknown").strip()
        # Simulate assistant direct response for Tier1 (optional)
        tier1_dialogue = [ {"role": "user", "content": ambiguous_instruction} ]
        assistant_response = None
        if target_call:
            # Create a simple confirmation message based on target_call
            if isinstance(target_call, str):
                assistant_response = target_call
            else:
                func = target_call.get("function", "action")
                params = target_call.get("parameters") or target_call.get("params") or {}
                if params:
                    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    assistant_response = f"OK, executed {func} with {params_str}."
                else:
                    assistant_response = f"OK, executed {func}."
        if assistant_response:
            tier1_dialogue.append({"role": "assistant", "content": assistant_response})
        tier1_data.append({
            "id": idx,
            "tier": 1,
            "dialogue": tier1_dialogue,
            "target_call": target_call,
            "meta": {
                "ambiguity_type": ambiguity_type,
                "protocol_compliant": True
            }
        })
        # Tier 2
        fuzz = fia.inject_fuzz_and_clarify(ambiguous_instruction, ambiguity_type)
        fuzzy_instruction = fuzz.get("fuzzy_instruction", ambiguous_instruction).strip()
        assistant_question = fuzz.get("assistant_question", "").strip()
        tier2_dialogue = [
            {"role": "user", "content": fuzzy_instruction},
            {"role": "assistant", "content": assistant_question}
        ]
        tier2_data.append({
            "id": idx,
            "tier": 2,
            "dialogue": tier2_dialogue,
            "target_call": target_call,
            "meta": {
                "ambiguity_type": ambiguity_type,
                "protocol_compliant": True
            }
        })
        # Tier 3
        dialogue_full = mea.expand_to_dialogue(fuzzy_instruction, assistant_question, target_call)
        tier3_data.append({
            "id": idx,
            "tier": 3,
            "dialogue": dialogue_full,
            "target_call": target_call,
            "meta": {
                "ambiguity_type": ambiguity_type,
                "protocol_compliant": True
            }
        })
    # Save to JSON files
    with open("tier1_single_turn.json", "w", encoding="utf-8") as f:
        json.dump(tier1_data, f, ensure_ascii=False, indent=2)
    with open("tier2_fuzzy_clarify.json", "w", encoding="utf-8") as f:
        json.dump(tier2_data, f, ensure_ascii=False, indent=2)
    with open("tier3_multi_turn.json", "w", encoding="utf-8") as f:
        json.dump(tier3_data, f, ensure_ascii=False, indent=2)
    print("Generation complete. Outputs: tier1_single_turn.json, tier2_fuzzy_clarify.json, tier3_multi_turn.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoControl-Bench Data Generation")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file with seed instructions.")
    parser.add_argument("--model", "-m", default="gpt-4", help="LLM model name (e.g. 'gpt-4', 'llama3-8B', 'Qwen-7B-Instruct').")
    args = parser.parse_args()
    generate_data(args.input, args.model)
