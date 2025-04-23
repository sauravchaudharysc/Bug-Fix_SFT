import json
import nltk
import subprocess
import tempfile
import os
import re
import javalang
from codebleu import calc_codebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict

nltk.download('punkt')

def normalize_code(code):
    return code.strip()

def compute_bleu(code_reference: str, code_generated: str) -> float:
    code_reference = normalize_code(code_reference)
    code_generated = normalize_code(code_generated)
    
    # Tokenizing the code into words
    ref_tokens = nltk.word_tokenize(code_reference)
    gen_tokens = nltk.word_tokenize(code_generated)
    
    # Smoothing function to handle cases where n-grams don't match
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)

def get_ast_node_types(java_code):
    """Extract AST node types from Java code using javalang"""
    try:
        # Parse the Java code
        tree = javalang.parse.parse(java_code)
        node_types = []
        
        # Walk through the tree and collect node types
        for path, node in tree:
            node_types.append(type(node).__name__)
        
        return node_types
    except Exception:
        # Return empty list if code cannot be parsed
        return []

def get_ast_node_bigrams(node_types):
    """Get bigrams of AST node types"""
    bigrams = []
    for i in range(len(node_types) - 1):
        bigrams.append((node_types[i], node_types[i+1]))
    return bigrams

def compute_ast_match(ref_code, gen_code):
    """Compute AST node match score"""
    ref_node_types = get_ast_node_types(ref_code)
    gen_node_types = get_ast_node_types(gen_code)
    
    if not ref_node_types or not gen_node_types:
        return 0.0
    
    ref_bigrams = get_ast_node_bigrams(ref_node_types)
    gen_bigrams = get_ast_node_bigrams(gen_node_types)
    
    if not ref_bigrams:
        return 0.0
    
    # Count matching bigrams
    matches = 0
    for bigram in gen_bigrams:
        if bigram in ref_bigrams:
            matches += 1
    
    if len(gen_bigrams) == 0:
        return 0.0
    return matches / len(gen_bigrams)

def extract_variable_names(java_code):
    """Extract variable names from Java code"""
    try:
        tree = javalang.parse.parse(java_code)
        variables = set()
        
        # Extract variable declarations
        for path, node in tree:
            if isinstance(node, javalang.tree.VariableDeclarator):
                variables.add(node.name)
                
            # Extract method parameters
            elif isinstance(node, javalang.tree.FormalParameter):
                variables.add(node.name)
        
        return variables
    except Exception:
        # Return empty set if code cannot be parsed
        return set()

def compute_dataflow_match(ref_code, gen_code):
    """Compute data flow match score based on variable usage"""
    ref_vars = extract_variable_names(ref_code)
    gen_vars = extract_variable_names(gen_code)
    
    if not ref_vars:
        return 0.0
    
    # Count matching variables
    matches = len(ref_vars.intersection(gen_vars))
    
    if len(gen_vars) == 0:
        return 0.0
    return matches / len(gen_vars)

def compute_codebleu(ref_code, gen_code, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25):
    """
    Compute CodeBLEU score for Java code
    alpha: weight for BLEU score
    beta: weight for AST match score
    gamma: weight for dataflow match score
    delta: weight for BLEU score with keywords
    """
    # Regular BLEU
    bleu_score = compute_bleu(ref_code, gen_code)
    
    # AST match
    ast_match_score = compute_ast_match(ref_code, gen_code)
    
    # Data flow match
    dataflow_match_score = compute_dataflow_match(ref_code, gen_code)
    
    # BLEU with Java keywords
    # Java keywords
    java_keywords = [
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 
        'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
        'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 
        'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 
        'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 
        'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 
        'try', 'void', 'volatile', 'while'
    ]
    
    def extract_keywords(code, keywords):
        tokens = nltk.word_tokenize(code)
        return [token for token in tokens if token in keywords]
    
    ref_keywords = extract_keywords(ref_code, java_keywords)
    gen_keywords = extract_keywords(gen_code, java_keywords)
    
    if not ref_keywords:
        keyword_bleu = 0.0
    else:
        smoothie = SmoothingFunction().method4
        keyword_bleu = sentence_bleu([ref_keywords], gen_keywords, smoothing_function=smoothie)
    
    # Weighted combination
    codebleu_score = alpha * bleu_score + beta * ast_match_score + gamma * dataflow_match_score + delta * keyword_bleu
    
    return codebleu_score, bleu_score, ast_match_score, dataflow_match_score, keyword_bleu

def check_javalang_installation():
    """Check if javalang is installed, if not install it"""
    try:
        import javalang
    except ImportError:
        print("Installing javalang package...")
        import pip
        pip.main(['install', 'javalang'])
        import javalang
    return True

def calculate_codebleu(code_reference, code_generated) -> float:
    result = calc_codebleu([code_reference], [code_generated], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return result

# Make sure javalang is installed
check_javalang_installation()

# Load the JSON files
with open("/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/FineTuneResults/results.json", "r") as f:
    result = json.load(f)
with open("/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/BugFix/test/fixed.json", "r") as f:
    fixed = json.load(f)

total_bleu = 0
total_codebleu = 0
total_codebleuscore = 0
total_ast_match = 0
total_dataflow_match = 0
total_keyword_bleu = 0
count = 0

# Store individual scores for analysis
all_scores = defaultdict(list)

# Process each code pair
for key in result:
    if key in fixed:
        fixed_code = fixed[key]
        generated_code = result[key]
        
        # Skip empty or invalid entries
        if not fixed_code or not generated_code:
            print(f"{key}: Empty or invalid code")
            continue
            
        try:
            # Compute regular BLEU score
            bleu = compute_bleu(fixed_code, generated_code)
            
            # Compute CodeBLEU score
            codebleu, bleu_component, ast_match, dataflow_match, keyword_bleu = compute_codebleu(fixed_code, generated_code)
            
            # Compute 
            codeBleu = calculate_codebleu(fixed_code,generated_code)

            # Update total scores
            total_bleu += bleu
            total_codebleu += codebleu
            total_codebleuscore += codeBleu['codebleu']
            total_ast_match += ast_match
            total_dataflow_match += dataflow_match
            total_keyword_bleu += keyword_bleu
            count += 1
           
            # Store individual scores
            all_scores[key] = {
                'bleu': bleu,
                'codebleu': codebleu,
                'ast_match': ast_match,
                'dataflow_match': dataflow_match,
                'keyword_bleu': keyword_bleu,
                'total_codebleuscore' : total_codebleuscore
            }
        except Exception as e:
            print(f"{key}: Error processing - {str(e)}")
    else:
        print(f"{key}: Missing in fixed.json")

# Calculate and display the average scores
if count > 0:
    avg_bleu = total_bleu / count
    avg_codebleu = total_codebleu / count
    avg_ast_match = total_ast_match / count
    avg_dataflow_match = total_dataflow_match / count
    avg_keyword_bleu = total_keyword_bleu / count
    avg_codebleuscore = total_codebleuscore / count
    
    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    print(f"Average CodeBLEU Score: {avg_codebleu:.4f}")
    print(f"Average AST Match Score: {avg_ast_match:.4f}")
    print(f"Average Dataflow Match Score: {avg_dataflow_match:.4f}")
    print(f"Average Keyword BLEU Score: {avg_keyword_bleu:.4f}")
    print(f"Average CodeBleu Score : {avg_codebleuscore:.4f}")
    
    # Export detailed results to a file
    with open("java_codebleu_detailed_results.json", "w") as f:
        json.dump({
            'average_scores': {
                'bleu': avg_bleu,
                'codebleu': avg_codebleu,
                'ast_match': avg_ast_match,
                'dataflow_match': avg_dataflow_match,
                'keyword_bleu': avg_keyword_bleu
            },
            'individual_scores': all_scores,
            'total_samples_processed': count
        }, f, indent=2)
    
    print("\nDetailed results saved to java_codebleu_detailed_results.json")
else:
    print("\nNo matching keys found to compute metrics.")