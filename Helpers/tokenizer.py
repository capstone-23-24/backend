import csv
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess

def openCSV(filePath):

    text = ""
    with open(filePath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row = processRow(', '.join(row))
            newRow = " ".join(row)
            text += newRow
            
            
    print(f'The joined string has been added to the CSV file: {text}')
    writeCSV(text)
    
    
def writeCSV(text):
    csv_filename = "output.csv"
    splitIndex = 32767
    
    if len(text) > splitIndex:
        
    
    with open(csv_filename, mode='w', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([text])
    
        
      
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
    openCSV('./stringified.csv')
    
    

if __name__ == "__main__":
    main()