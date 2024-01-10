import csv
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

def openCSV(filePath):
    column1 = 'case_part1' 
    column2 = 'case_part2'
    
    combined_text = []
    with open(filePath, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        if column1 not in csv_reader.fieldnames or column2 not in csv_reader.fieldnames:
            print("no such column found in csv")
        
        for row in csv_reader:
            combined_text = f"{row[column1]} {row[column2]}"
            rawText = processRow(combined_text)
            updateCSV(" ".join(rawText))
    
def updateCSV(text):
    csv_filename = "SMOutput.csv"
    splitIndex = 32767
    
    case_part1 = text[:splitIndex+ 1]
    case_part2 = text[splitIndex + 1:]
    
    with open(csv_filename, mode='a', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(["case_part1", "case_part2"])  
            
        csv_writer.writerow([case_part1, case_part2])
                
def processRow(input_text):
    remove_stopwords(input_text)
    tokens = simple_preprocess(input_text, deacc=True)
    
    # # You can add more custom stop words if needed
    custom_stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    
    filtered_tokens = [word for word in tokens if word.lower() not in custom_stop_words]

    return filtered_tokens
          
def main():
    # this is the test case
    openCSV('./stringified.csv')
    
    """
    _summary_
    
    Use this function to iterate through the text data
    in a populated csv to pupolate the SMOutput.csv with the process results
    
    Open Csv function earlier is just the test function to handle a single case
    should be extended iterate through full csv. 
    """
    

if __name__ == "__main__":
    main()