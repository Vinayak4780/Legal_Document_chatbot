import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.replace('\n\n', '\n') + "\n"
    return text

def remove_extra_whitespaces(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = re.sub(r'\s+', ' ', line).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    merged_lines = []
    current_line = ""
    
    for line in cleaned_lines:
        if re.match(r'^\d+\.', line) or line.isupper() and len(line.split()) <= 4 or re.match(r'^[A-Z]\.', line):
            if current_line:
                merged_lines.append(current_line)
                current_line = ""
            merged_lines.append(line)
        else:
            if current_line and not current_line.rstrip().endswith(('.', '!', '?', ':', ';')):
                current_line += " " + line
            else:
                if current_line:
                    merged_lines.append(current_line)
                current_line = line
    
    if current_line:
        merged_lines.append(current_line)
    
    return '\n'.join(merged_lines)

def apply_preprocessing_to_sentence(sentence):
    if not sentence.strip():
        return sentence
    
    processed = sentence.strip()
    
    if '"' in processed:
        parts = []
        current_pos = 0
        quote_positions = [i for i, char in enumerate(processed) if char == '"']
        
        if len(quote_positions) % 2 == 0:
            for i in range(0, len(quote_positions), 2):
                start_quote = quote_positions[i]
                end_quote = quote_positions[i + 1]
                
                if start_quote > current_pos:
                    text_before = processed[current_pos:start_quote]
                    parts.append(text_before.lower())
                
                quoted_text = processed[start_quote:end_quote + 1]
                parts.append(quoted_text)
                current_pos = end_quote + 1
            
            if current_pos < len(processed):
                text_after = processed[current_pos:]
                parts.append(text_after.lower())
            
            processed = ''.join(parts)
        else:
            processed = processed.lower()
    else:
        processed = processed.lower()
    
    processed = re.sub(r'[^\w\s.,():;"\'-]', ' ', processed)
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    if '"' not in processed:
        words = word_tokenize(processed)
        words = [re.sub(r"[,:;]", "", word) for word in words]
        filtered_words = [word for word in words if word.lower() not in stop_words or word.isdigit() or word in '.,()"\'-']
        lemmatized_words = []
        for word in filtered_words:
            if word.isalpha():
                lemmatized_words.append(lemmatizer.lemmatize(word))
            else:
                lemmatized_words.append(word)
        processed = ' '.join(lemmatized_words)
    else:
        result_parts = []
        current_pos = 0
        quote_positions = [i for i, char in enumerate(processed) if char == '"']
        
        if len(quote_positions) % 2 == 0:
            for i in range(0, len(quote_positions), 2):
                start_quote = quote_positions[i]
                end_quote = quote_positions[i + 1]
                
                if start_quote > current_pos:
                    text_before = processed[current_pos:start_quote].strip()
                    words = word_tokenize(text_before)
                    words = [re.sub(r"[,:;]", "", word) for word in words]
                    filtered_words = [word for word in words if word.lower() not in stop_words or word.isdigit() or word in '.,()"\'-']
                    lemmatized_words = []
                    for word in filtered_words:
                        if word.isalpha():
                            lemmatized_words.append(lemmatizer.lemmatize(word))
                        else:
                            lemmatized_words.append(word)
                    result_parts.append(' '.join(lemmatized_words))
                
                quoted_text = processed[start_quote:end_quote + 1]
                result_parts.append(quoted_text)
                current_pos = end_quote + 1
                
            if current_pos < len(processed):
                text_after = processed[current_pos:].strip()
                words = word_tokenize(text_after)
                words = [re.sub(r"[,:;]", "", word) for word in words]
                filtered_words = [word for word in words if word.lower() not in stop_words or word.isdigit() or word in '.,()"\'-']
                lemmatized_words = []
                for word in filtered_words:
                    if word.isalpha():
                        lemmatized_words.append(lemmatizer.lemmatize(word))
                    else:
                        lemmatized_words.append(word)
                result_parts.append(' '.join(lemmatized_words))
            
            processed = ' '.join(result_parts)
        else:
            words = word_tokenize(processed)
            words = [re.sub(r"[,:;]", "", word) for word in words]
            filtered_words = [word for word in words if word.lower() not in stop_words or word.isdigit() or word in '.,()"\'-']
            lemmatized_words = []
            for word in filtered_words:
                if word.isalpha():
                    lemmatized_words.append(lemmatizer.lemmatize(word))
                else:
                    lemmatized_words.append(word)
            processed = ' '.join(lemmatized_words)
    
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    if processed and not processed.endswith('.'):
        processed += '.'
    
    return processed

def format_topics_and_structure(text):
    lines = text.split('\n')
    structured_content = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        if i == 0 and line:
            structured_content.append(line.upper())
            structured_content.append("")
            
        elif re.match(r'^\d+\.', line):
            structured_content.append("")
            structured_content.append(line.upper())
            
        elif (len(line.split()) <= 4 and 
              any(keyword in line.lower() for keyword in ['introduction', 'ebay', 'using', 'vehicle', 'policy', 'fee', 'listing', 'purchase', 'international', 'content', 'notice', 'hold', 'authorization', 'additional', 'payment', 'disclaimer', 'release', 'indemnity', 'legal', 'general'])):
            
            structured_content.append("")
            structured_content.append(line.upper())
            
        elif re.match(r'^[A-Z]\.', line):
            structured_content.append(f"  {line.upper()}")
            
        else:
            processed_line = apply_preprocessing_to_sentence(line)
            if not processed_line.strip().endswith('.'):
                processed_line = processed_line.strip() + '.'
            structured_content.append(processed_line)
    
    return '\n'.join(structured_content)

def preprocess_pdf_file(pdf_path, output_path=None):
    text = extract_text_from_pdf(pdf_path)
    text = remove_extra_whitespaces(text)
    text = format_topics_and_structure(text)
    text = remove_extra_whitespaces(text)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    return text

def process_pdf_from_data_folder(pdf_filename, output_filename=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    pdf_path = os.path.join(parent_dir, 'data', pdf_filename)
    
    if not os.path.exists(pdf_path):
        return None
    
    if output_filename:
        output_path = os.path.join(parent_dir, 'data', output_filename)
    else:
        output_path = os.path.join(parent_dir, 'data', f"preprocessed_{pdf_filename.replace('.pdf', '.txt')}")
    
    return preprocess_pdf_file(pdf_path, output_path)

def main():
    pdf_filename = "AI Training Document.pdf"
    result = process_pdf_from_data_folder(pdf_filename)
    return result

if __name__ == "__main__":
    main()