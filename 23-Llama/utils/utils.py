#import langchain.llms import OpenAI
from pypdf import PdfReader
#from langchain.llms.openai import OpenAI
import pandas as pd
import re
import replicate
import whisper
#from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers
import langchain
from sklearn.model_selection import train_test_split

def get_transformer():
    llm = langchain.llms.CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens':256, 
                                'temperature':0.01})
    return llm

def get_prompt_template(template):
    prompt = langchain.prompts.PromptTemplate(input_variables=['instruction', 'input'],
                            template=template)
    return prompt

#Splitting the data into train & test
def split_train_test_data(df_sample):
    # Split into training and testing sets
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0)
    print(len(sentences_train))
    return sentences_train, sentences_test, labels_train, labels_test

#Get the accuracy score on test data
def get_score(svm_classifier,sentences_test,labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score

def extract_text_from_audio(file):
    print(whisper.__version__)
    model = whisper.load_model("base")
    fpath = "audio/" + file
    print("file=", fpath)
    result = model.transcribe(fpath)
    print(result["text"])
    return result["text"]

# extract information from pdf
def get_pdf_text(pdf_doc):
    print(pdf_doc)
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#Read dataset for model creation
def read_data_set(data):
    df = pd.read_csv(data,delimiter=',', header=None)  
    return df

# function to extract data from text using LLM
def extract_data(pages_data):

    template = """Extract all the following values : invoice no., Description, Quantity, date, 
        Unit price , Amount, Total, email, phone number and address from this data: {pages}

        Expected output: remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair','Quantity': '2','Date': '5/4/2023','Unit price': '1100.00','Amount': '2200.00','Total': '2200.00','Email': 'Santoshvarma0988@gmail.com','Phone number': '9999999999','Address': 'Mumbai, India'}}
        """
    # template = """Extract all the following values : invoice no., Description, Quantity, date,
    # Unit price, Amount, Total, email, phone number and address from this data: {pages}
    
    # Expected output: remove any dollar symbols {{'Invoice no.': '1001329','Description': 'Office Chair', 'Quantity': '2',
    # 'Date': '5/4/2023', 'Unit Price': '1100.00', 'Amount': '2200.00', 'Total': '2200.00', 'Email': 'Santoshvarma0988@gmail.com',
    # 'Phone number': '9999999999', 'Address': 'Mumbai India'}}
    # """

    prompt_template = get_prompt_template(template)

    #llm = OpenAI(temperature=0.7)
    #full_response = llm(promptTemplate.format(pages=pages_data))

    # output = replicate.run('replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1', 
    #                        input={"prompt":prompt_template.format(pages=pages_data) ,
    #                               "temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})

    llm = get_transformer()

    output = llm(prompt_template.format(pages=pages_data))
    full_response = ''
    for item in output:
        full_response += item

    print(full_response)
    return full_response

# de
def create_docs(user_pdf_list):
    df = pd.DataFrame({'Invoice no.': pd.Series(dtype='str'),
                       'Description': pd.Series(dtype='str'),
                       'Date': pd.Series(dtype='str'),
                       'Unit price': pd.Series(dtype='str'),
                       'Amount': pd.Series(dtype='int'),
                       'Total': pd.Series(dtype='str'),
                       'Email': pd.Series(dtype='str'),
	                   'Phone number': pd.Series(dtype='str'),
                       'Address': pd.Series(dtype='str')
                       })
    for file_name in user_pdf_list:
        print(file_name)
        raw_data = get_pdf_text(file_name)
        llm_extracted_text = extract_data(raw_data)
        pattern = r'{(.+)}'
        match = re.search(pattern, llm_extracted_text, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            print(extract_data)
            try:
                data_dict = eval('{' + extracted_text + '}')
                print(data_dict)
            except:
                print("exception")
        else:
            print("No match found")
        
        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        print("DONE............")
    df.head()
    return df
