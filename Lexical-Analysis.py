import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def load_data(file_path):
    """
    Load data from a file (.csv or .txt).

    Args:
        file_path: Path to the input file.

    Returns:
        Pandas DataFrame with `text_id` and `content` columns.
    """
    # Check file extension to determine the loading method
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        df = pd.DataFrame({'text_id': range(1, len(lines) + 1), 'content': lines})
    else:
        raise ValueError("Unsupported file format. Only .csv and .txt files are supported.")
    
    if 'content' not in df.columns:
        raise ValueError("The input file must contain a 'content' column or be a text file.")
    
    return df

# Data preprocessing
def preprocess_data(df):
    """
    Preprocesses the text data by converting to lowercase and removing non-alphanumeric characters.

    Args:
        df: Pandas DataFrame with a `content` column.

    Returns:
        Updated DataFrame with a `cleaned_content` column.
    """
    df['cleaned_content'] = df['content'].str.lower()
    df['cleaned_content'] = df['cleaned_content'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', str(x)))
    return df

def retrieve_related_expressions(reference_expressions, data_frame):
    """
    Retrieves related expressions based on a list of user-provided reference expressions.

    Args:
        reference_expressions: A list of regex patterns provided by the user.
        data_frame: Pandas DataFrame with a column `cleaned_content`.

    Returns:
        A structured DataFrame with results.
    """
    results = []
    for expression in reference_expressions:
        try:
            # Compile the regex expression
            compiled_regex = re.compile(expression)
            for _, row in data_frame.iterrows():
                # Find all matches in the cleaned content
                matches = compiled_regex.findall(row['cleaned_content'])
                if matches:
                    results.append({
                        'text_id': row['text_id'],
                        'original_content': row['content'],
                        'matched_expressions': ', '.join(matches),  # Join matches for readability
                        'reference_expression': expression
                    })
        except re.error as e:
            print(f"Invalid regular expression: {e}")

    # Return results as a DataFrame
    return pd.DataFrame(results)

def main():
    # Specify the file path directly in the code
    path = "C:/Users/vhanj/Downloads/Computer.txt"  # Update this to the correct path for your file (CSV or TXT)

    try:
        df = load_data(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Preprocess the data
    df = preprocess_data(df)
    
    # Export preprocessed data to TXT
    preprocessed_output_file = "preprocessed_data.txt"
    df[['text_id', 'cleaned_content']].to_csv(preprocessed_output_file, sep='\t', index=False, header=True)
    print(f"Preprocessed data exported to {preprocessed_output_file}")

    # Prompt the user for regex patterns
    user_inputs = input("Enter the patterns to search for, separated by commas (e.g., 'ing, pattern, ed'): ").split(',')
    reference_expressions = [rf"\b\w*{pattern.strip()}\w*\b" for pattern in user_inputs]

    # Retrieve related expressions
    related_expressions_df = retrieve_related_expressions(reference_expressions, df)

    # Export results to a CSV file
    output_file = "related_expressions_results.csv"
    related_expressions_df.to_csv(output_file, index=False)

    print(f"Related expressions exported to {output_file}")

# Entry point
if __name__ == "__main__":
    main()
