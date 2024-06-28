__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# import spacy
# import spacy_streamlit
# from wordcloud import WordCloud
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json


#####     CONSTANTS     #####
OPENAI_APIKEY = st.secrets["openai_apikey"]
EMBEDDING_MODEL = 'text-embedding-3-large'  # 'text-embedding-3-small'
# SPACY_MODEL = spacy.load('en_core_web_sm') # 'en_core_web_lg'
SDG = ["No Poverty", "Zero Hunger", "Good Health and Well-being", "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy", "Decent Work and Economic Growth", "Industry, Innovation and Infrastructure", "Reduced Inequalities", "Sustainable Cities and Economies", "Responsible Consumption and Production", "Climate Action", "Life Below Water", "Life on Land", "Peace, Justice and Strong Institutions", "Partnership for the Goals"]
SDG_EDITED = []
TAGALOG_STOP_WORDS = set("applause nga ug eh yun yan yung kasi ko akin aking ako alin am amin aming ang ano anumang apat at atin ating ay bababa bago bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa ginawa ginawang gumawa gusto habang hanggang hindi huwag iba ibaba ibabaw ibig ikaw ilagay ilalim ilan inyong isa isang itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang kanino kanya kanyang kapag kapwa karamihan katiyakan katulad kaya kaysa ko kong kulang kumuha kung laban lahat lamang likod lima maaari maaaring maging mahusay makita marami marapat masyado may mayroon mga minsan mismo mula muli na nabanggit naging nagkaroon nais nakita namin napaka narito nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa paggawa pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan pareho pataas pero pumunta pumupunta sa saan sabi sabihin sarili sila sino siya tatlo tayo tulad tungkol una walang ba eh kasi lang mo naman opo po si talaga yung".split())
APP_NAME = 'SONAS: Semantic OpenAI NLP Assistant System'
APP_DESC = ' @Team Dyn4mix'
COLOR_BLUE = '#0038a8'
COLOR_RED = '#ce1126'
COLOR_LIGHTER_RED = "#FFF2F2"
COLOR_GRAY = '#f8f8f8'

# Load SONA dataset
df = pd.read_csv("data/sonas.csv")
sonas_df = pd.read_csv("data/edited_sonas_with_summary_and_score.csv")

#####     FUNCTIONS     #####
def get_openai_client():
    """Function to create OpenAI Client"""
    openai.api_key = OPENAI_APIKEY
    client = OpenAI(api_key=OPENAI_APIKEY)
    return client


def init_chroma_db(collection_name, db_path='https://github.com/babybunso/dsf-s3/blob/master/sonas_new.db'):
    """Function to initialize chromadb client"""
    # Create a Chroma Client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Create an embedding function
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_APIKEY, model_name=EMBEDDING_MODEL)

    # Create a collection
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    return collection


def general_semantic_search(Q, k=3, collection=None):
    """Function to query the whole collection"""
    results = collection.query(
        query_texts=[Q], # Chroma will embed this for you
        n_results=k, # how many results to return,
    )
    return results


def specific_semantic_search(Q, k=3, collection=None, metadata_key = "", meta_val = ""):
    """Function to query a subset of the collection (based on a metadata)"""
    results = collection.query(
        query_texts=[Q], # Chroma will embed this for you
        n_results=k, # how many results to return,
        where={f"{metadata_key}": f"{meta_val}"} # specific data only
    )
    return results


def generate_response(task, prompt, llm):
    """Function to generate a response from the LLM given a specific task and user prompt"""
    response = llm.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': f"Perform the specified task: {task}"},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response.choices[0].message.content
    

def generate_summary(doc, llm, word_limit=None):
    """Function to ask the LLM to create a summary"""
    task = "Text Summarization"
    prompt = "Summarize this document"
    if word_limit:
        prompt += f" in {word_limit} words"

    prompt += f":\n\n{doc}"
    response = generate_response(task, prompt, llm)
    return response


def generate_translation(doc, llm, lang="Tagalog"):
    """Function to translate a document from English to Language of Choice"""
    task = 'Text Translation'
    prompt = f"Translate this document from English to {lang}:\n\n{doc}"
    response = generate_response(task, prompt, llm)

    return response


def generate_key_themes(doc, llm, top_k=5):
    """Function to get the main points/themes of a document"""
    task = "Main points or theme extraction or topic modeling"
    prompt = f"Extract and list the top {top_k} main themes in this document:\n\n{doc}"
    # prompt = f"Extract and list the top {top_k} main themes in this document:\n\n....{doc}...."
    response = generate_response(task, prompt, llm)
    return response

def generate_key_sdgthemes(doc, llm, top_k=10):
    """Function to get the main points/themes of a document"""
    task = "Main points or theme extraction or topic modeling"
    prompt = f"""
    Extract and list the top {top_k} main 'Sustainable Development Goals': {', '.join(SDG)} in this document:\n\n{doc}

    The output should be list format like:
        [No Poverty, Zero Hunger, Good Health, ...]
    """
    print(task + "\n\n" + prompt)
    response = generate_response(task, prompt, llm)
    return response

def summarize_single_SONA(collection=None, title="", llm=None):
    """Function to get the summary of a whole SONA"""
    lst_of_docs = collection.get(where={"title": title}, include=['documents'])['documents']
    lst_of_summaries = []

    # Get the summary of each doc and put it in the list
    for doc in lst_of_docs:
        summary = generate_summary(doc, llm)
        lst_of_summaries.append(summary)

    # Ask the LLM to create a summary based on the list of summaries
    task = "Text Summarization"
    prompt = f"From a list of summaries, create a one paragraph comprehensive summary. Make sure to include statistcs, action points, and policies as much as possible:\n\n{lst_of_summaries}"
    response = generate_response(task, prompt, llm)

    return response


def generate_SONA_summary(df, llm):
    task = "Text Summarization"
    documents = df['text'].to_list()
    doc_input = ""

    for i in range(len(df)):
        doc_input += f"""
        Document {i} Content: {documents[i]}
        """
    
    prompt = f"""
    You are an unbiased, fair, honest, intelligent, and an expert jounalist-researcher that is very knowledgeable in different domain of expertise encompassing investigative journalism.
    
    You are not representing any party or organization and you would treat the documents as research materials for intelligent context searching for you to be able to report the similiraties and differences of what has been written in those documents.

    Your task is to generate a comprehensive one paragraph summary about the whole research document in this list: {doc_input}

    As much as possible, include specific statistics, policies, plans, or actions.
    """
    response = generate_response(task, prompt, llm)
    return response


def generate_doc_summary(Q, doc, llm):
    task = "Keypoints"
    prompt = f"""Identify the keypoints and its description that are inline with the topic {Q} from the document:\n\n{doc}

    The output should be in a bullet point format. Make sure to include the description and the groups affected of the keypoint as well. An example is:
        - <Keypoint 1 Title>: <Description of Keypoint 1 and groups affected>
        - <Keypoint 2 Title>: <Description of Keypoint 2 and groups affected>
    """
    response = generate_response(task, prompt, llm)
    return response


##### STYLED COMPONENTS #####
# set universal styles
html_styles = f"""
    <style>
        h3 {{
            color: {COLOR_BLUE};
        }}

        p {{
            font-size: 1.125rem;
            text-align: justify;
        }}

        .st-emotion-cache-1v0mbdj img {{
            position: relative;
            border-radius: 50%;
        }}

        .st-emotion-cache-1v0mbdj {{
            border-radius: 50%;
            border: 5px solid transparent;
            background-image: linear-gradient(white, white), 
                            linear-gradient(to right, {COLOR_RED}, {COLOR_BLUE});
            background-origin: border-box;
            background-clip: content-box, border-box;
        }}

        .bolded {{
            font-weight: 900;
        }}

        .italicized {{
            font-style: italic;
        }}

        .tabbed {{
            margin-left: 1.75rem;
            margin-top: 0;
        }}
    </style>
"""

app_desc_html = f"""
<style>
    .code-like {{
        background-color: {COLOR_GRAY};
        display: inline;
        margin: 0;
    }}

    .team-name {{
        margin: 1rem;
        font-size: 1rem;
        color: green;
        display: inline;
    }}
</style>
<div class='code-like'>
    <p class='team-name'>{APP_DESC}</p>
</div>
"""

homepage_html = f"""
<style>
</style>
<h3>About the Project</h3>
<p>Welcome to the SONA Analysis Tool! This app is designed to make State of the Nation Addresses (SONA) more accessible and informative for everyone. Explore concise summaries of SONA speeches, analyze presidential themes and focus areas, and dive into specific topics like healthcare, education, and the economy.</p>
<p>Whether you're a student, researcher, policymaker, or simply a curious citizen, our user-friendly filters and insights will help you better understand the priorities and achievements of different administrations.</p>

<h3>Page Descriptions</h3>
<h4>Per SONA Summarizer</h4>
<p>SONA Summarizer offers concise summaries of SONA speeches, highlighting key points and themes. Quickly grasp the main messages delivered by each president in their annual addresses</p>

<h4>Query Analysis</h4>
<p>This page allows you to explore specific topics within SONA speeches, such as healthcare, education, the economy, national security, environmental issues, infrastructure development, transportation, agriculture, and governance reforms. You can also examine how each Sustainable Development Goal (SDG) is addressed in the SONA speeches.  The personalized filters enable you to focus your analysis on the issues most relevant to your interests, providing a comprehensive understanding of how these topics have been addressed in different SONA speeches over time.</p>
"""

def keywords_line(keywords_lst):
    """Function to style the keywords list (below the title)"""

    keyword_str = ", ".join(keywords_lst)
    
    keyword_html = f"""
    <style>
        .keywords_lst {{
            margin-bottom: 2px;
            font-size: 1rem;
        }}
    </style>
    <p class="keywords_lst">Keywords: {keyword_str}</p>
    """
    st.markdown(keyword_html, unsafe_allow_html=True)


def summary_cards(eng_sum, fil_sum, bis_sum):
    """Function to style the summary columns"""
    
    split_col_html = f"""
    <style>
        .summary-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        div.left, div.mid, div.right {{
            flex: 32%;
            padding: 15px 30px;
        }}

        div.left {{
            border: 1px, solid, {COLOR_LIGHTER_RED};
            border-radius: 20px;
            background-color: {COLOR_LIGHTER_RED};
        }}

        div.mid, div.right {{
            border: 1px, solid, {COLOR_GRAY};
            border-radius: 20px;
            background-color: {COLOR_GRAY};
        }}
    </style>

    <div class="summary-container">
        <div class="left">
            <h3><span class='bolded'>English</span> Summary</h3>
            <p>{eng_sum}</p>
        </div>
        <div class="mid">
            <h3><span class='bolded'>Filipino</span> Summary</h3>
            <p>{fil_sum}</p>
        </div>
        <div class="right">
            <h3><span class='bolded'>Bisaya</span> Summary</h3>
            <p>{bis_sum}</p>
        </div>
    </div>
    """
    
    st.markdown(split_col_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def summary_cards_v2(eng_sum, fil_sum, bis_sum):
    st.markdown("<br>", unsafe_allow_html=True)

    eng_html = f"""
    <style>
        .eng-container {{
            border: 1px, solid, {COLOR_GRAY};
            border-radius: 20px;
            background-color: {COLOR_GRAY};
            padding: 15px 30px;
        }}
    </style>
    
    <div class="eng-container">
        <h3><span class='bolded'>English</span> Summary</h3>
        <p>{eng_sum}</p>
    </div>
    """

    other_translations_html = f"""
    <style>
        .summary-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        div.left, div.right {{
            flex: 40%;
            padding: 15px 30px;
        }}
    </style>

    <div class="summary-container">
        <div class="left">
            <h3><span class='bolded'>Filipino</span> Summary</h3>
            <p>{fil_sum}</p>
        </div>
        <div class="right">
            <h3><span class='bolded'>Bisaya</span> Summary</h3>
            <p>{bis_sum}</p>
        </div>
    </div>
    <br>
    """

    st.markdown(eng_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Expand to view the summary in Filipino and Bisaya", expanded=False):
        st.markdown(other_translations_html, unsafe_allow_html=True)


def recommender_output(recommendation):
    recommend_html = f"""
    <h3>Recommendation</h3>
    <p>{recommendation}</p>
    """
    st.markdown(recommend_html, unsafe_allow_html=True)


def actions_layout(point_lst, desc_lst, stat_lst, policies, class_lst, class_exp, lang):
    """Function to style the action + implication section"""
    st.markdown("<br>", unsafe_allow_html=True)
    imp_header = ""
    key_stat = "Key Statistic"
    pol = "Policy"
    cur_progress = "Current Progress"

    if lang == "E":
        st.subheader("Policies and Plans")
        st.markdown(f"<p class=italicized>The following were the key policies and plans mentioned by president {presi}</p>", unsafe_allow_html=True)
        imp_header = "More Information about"
    elif lang == "F":
        st.subheader("Mga Plano at Patakaran")
        st.markdown(f"<p class=italicized>Ang mga sumusunod ay ang mga patakaran at plano na binanggit ni pangulong {presi}</p>", unsafe_allow_html=True)
        imp_header = "Karagdagang Impormasyon Tungkol sa"
        key_stat = "Pangunahing Istatistika"
        pol = "Patakaran"
        cur_progress = "Kasalukuyang Pagsulong"
    else:
        st.subheader("Mga Polisiya ug Plano")
        st.markdown(f"<p class=italicized>Ang mosunod mao ang mahinungdanong mga polisiya ug mga plano nga gihisgutan sa presidente {presi}</p>", unsafe_allow_html=True)
        imp_header = "Dugang impormasyon mahitungod sa"
        key_stat = "Pangunang Estadistika"
        pol = "Polisiya"
        cur_progress = "Kasamtangang Pag-uswag"

    for i in range(len(point_lst)):
        actions_html = f"""
        <h4>{i+1}. {point_lst[i]}</h4>
        <p>{desc_lst[i]}</p>
        """

        more_info = f"""
        <style>
            .summ-card {{
                margin: 0.5rem;
            }}
            .progress {{
                margin-bottom: 0px;
            }}
        </style>
        <p class='summ-card'><span class='bolded'>{key_stat}: </span>{stat_lst[i]}</p>
        <p class='summ-card'><span class='bolded'>{pol}: </span>{policies[i]}</p>
        <p class='progress summ-card'><span class='bolded'>{cur_progress}: </span>{class_lst[i]}</p>
        <div class='summ-card'><p class='tabbed'>{class_exp[i]}</p></div>
        """

        st.markdown(actions_html, unsafe_allow_html=True)

        with st.expander(f"{imp_header} {point_lst[i]}", expanded=False):
            # st.write('hello')
            st.markdown(more_info, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def query_results(title, sources, policies):
    for i in range(len(title)):
        per_sona_result = f"""
        
        <h3>{title[i]}</h3>
        <p class='italicized'>{sources[i]}</p>
        """

        st.markdown(per_sona_result, unsafe_allow_html=True)

        policy = policies[i].split("- ")

        with st.expander("View Related Policies", expanded=False):
            for pol in policy:
                query_html = f"""
                <p>{pol.strip()}</p>
                """
                st.markdown(query_html, unsafe_allow_html=True)
        


def president_card(name, img_path, info, themes):
    """Function to style the president info section"""
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,2], gap='large')

    presi_info_html = f"""
    <style>
        .presi-info, .presi-info p {{
            font-size: 20px;
        }}
        
        .presi-info .info-header {{
            margin-bottom: 0;
        }}
        
    </style>

    <p class="presi-info italicized">{info["birth"]}</p>
    <p class="presi-info"><span class="bolded info-header">Presidential Term:</span> {info['term']} ({info['no']} President)</p>
    <div class="presi-info">
        <p class="bolded info-header">Primary Theme of SONAs <i>(SDG Goals)</i>:</p>
        <p>{themes}</p>
    </div>
    """

    with col1:
        st.image(img_path, use_column_width="always")
    with col2:
        st.title(name)
        st.markdown(presi_info_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


#####     MAIN SITE     #####

# Initialize chroma db
collection = init_chroma_db(collection_name="sonas", db_path="https://github.com/babybunso/dsf-s3/blob/master/sonas_new.db")

# Initialize OpenAI client
llm = get_openai_client()

# Create streamlit app
st.set_page_config(layout='wide')
st.title(APP_NAME)
st.markdown(app_desc_html, unsafe_allow_html=True)
st.markdown(html_styles, unsafe_allow_html=True)

options = st.sidebar.radio("", ["Home", "Per SONA Summarizer", "Query Analysis", "Presidential Analysis"])

if options == "Home":
    st.write('___')
    st.markdown(homepage_html, unsafe_allow_html=True)


elif options == "Per SONA Summarizer":
    st.write('___')
    title = st.selectbox('Choose a SONA to Summarize', df['title.cleaned'])
    if title:
        # Get the content dedicated to the specific SONA
        current_sona = sonas_df[sonas_df['title.cleaned'] == title]
        # Get info about the selected SONA
        presi = list(current_sona['president'])[0].strip()
        link = list(current_sona['url'])[0].strip()
        # keywords = ["test 1", "test 2"]

        st.header(title)
        # keywords_line(keywords)
        st.caption(link)

        # Load file
        json_file = list(current_sona['json_summary'])[0]
        dict_file = json.loads(json_file)

        # Get summaries
        eng_sum = dict_file['Executive Summary']
        fil_sum = generate_translation(eng_sum, llm, "Tagalog")
        bis_sum = generate_translation(eng_sum, llm, "Bisaya")
        summary_cards_v2(eng_sum, fil_sum, bis_sum)

        # Get keypoints
        eng_point_lst = list(dict_file['Keypoints'].keys())
        eng_desc_lst = [dict_file['Keypoints'][point]['Context'] for point in eng_point_lst]
        eng_stat_lst = [dict_file['Keypoints'][point]['Descriptive Statistics'] for point in eng_point_lst]
        eng_policies = [dict_file["Keypoints"][point]['Policies'] for point in eng_point_lst]
        eng_classification = [dict_file["Keypoints"][point]['Classification'] for point in eng_point_lst]
        eng_class_exp = [dict_file["Keypoints"][point]['Classification Explanation'] for point in eng_point_lst]

        # Get Recommendation
        recommendation = dict_file['Recommendations']
        
        with st.expander("Expand to change the language", expanded=False):
            lang = st.radio(label="Choose a language", options=['English', 'Filipino', 'Bisaya'], index=0)


        if lang == "English":
            actions_layout(eng_point_lst, eng_desc_lst, eng_stat_lst, eng_policies, eng_classification, eng_class_exp, "E")
        elif lang == "Filipino":
            fil_point_lst = [generate_translation(eng_ver, llm, "Tagalog") for eng_ver in eng_point_lst]
            fil_desc_lst = [generate_translation(eng_ver, llm, "Tagalog") for eng_ver in eng_desc_lst]
            fil_stat_lst = [generate_translation(eng_ver, llm, "Tagalog") for eng_ver in eng_stat_lst]
            fil_policies = [generate_translation(eng_ver, llm, "Tagalog") for eng_ver in eng_policies]
            fil_classification = [generate_translation(eng_ver, llm, "Tagalog") for eng_ver in eng_classification]
            fil_class_exp = [generate_translation(eng_ver, llm, "Tagalog") for eng_ver in eng_class_exp]
            actions_layout(fil_point_lst, fil_desc_lst, fil_stat_lst, fil_policies, fil_classification, fil_class_exp, "F")
        else:
            bis_point_lst = [generate_translation(eng_ver, llm, "Bisaya") for eng_ver in eng_point_lst]
            bis_desc_lst = [generate_translation(eng_ver, llm, "Bisaya") for eng_ver in eng_desc_lst]
            bis_stat_lst = [generate_translation(eng_ver, llm, "Bisaya") for eng_ver in eng_stat_lst]
            bis_policies = [generate_translation(eng_ver, llm, "Bisaya") for eng_ver in eng_policies]
            bis_classification = [generate_translation(eng_ver, llm, "Bisaya") for eng_ver in eng_classification]
            bis_class_exp = [generate_translation(eng_ver, llm, "Bisaya") for eng_ver in eng_class_exp]
            actions_layout(bis_point_lst, bis_desc_lst, bis_stat_lst, bis_policies, bis_classification, bis_class_exp, "B")
    
        recommender_output(recommendation)


elif options == "Presidential Analysis":
    presi_list = list(df['president'].unique())
    presi_imgs = ["bbm.png", "duterte.png", "pnoy.jpg", "gma.jpg", "erap.jpg"]

    # List containing birth to death and presidential term
    presi_info = [
        {"birth": "September 13, 1957 - Present", "term": "June 30, 2022 - Present", "no": "17th"}, # bbm

        {"birth": "March 28, 1945 - Present", "term": "June 30, 2016 – June 30, 2022", "no": "16th"}, # duterte

        {"birth": "February 8, 1960 - June 24, 2021", "term": "June 30, 2010 – June 30, 2016", "no": "15th"}, # pnoy

        {"birth": "April 5, 1947 - Present", "term": "January 20, 2001 - June 30, 2010", "no": "14th"}, # gma

        {"birth": "April 19, 1937 - Present", "term": "June 30, 1998 – January 20, 2001", "no": "13th"}, # erap
    ]

    st.write('___')
    presi = st.selectbox(label="President Selector", options=presi_list, index=None, placeholder="Select a president")

    if presi:
        presi_idx = presi_list.index(presi)
        img_link = f"imgs/{presi_imgs[presi_idx]}" # Get the image
        info = presi_info[presi_idx]

         # Get the content dedicated to the specific SONA
        current_sonas = sonas_df[sonas_df['president'] == presi  ]
        # Load file
        json_files = list(current_sonas['json_summary'])
        theme_recos = ''
        for json_file in json_files:
            dict_file = json.loads(json_file)
            # # Get summaries
            eng_sum = dict_file['Executive Summary']
            # # Get Recommendation
            recommendation = dict_file['Recommendations']
            theme_recos += eng_sum + '\n' + recommendation + '\n\n\n'

        theme = generate_key_sdgthemes(theme_recos, llm, top_k=10)
        theme = theme.replace("[", "").replace("]", "").replace('"', "")

        president_card(presi, img_link, info, theme)


elif options == "Query Analysis":
    st.write('___')
    results = ""
    with st.expander("Specific Query Search", expanded=True):
            query_type = st.radio("Select a Query Type", ['Select Sustainable Development Goal Topic(s)', 'Ask a Question?'])
            
            if query_type == "Select Sustainable Development Goal Topic(s)":
                SDGQ = st.multiselect('SDG Goals:', SDG)
                Q = ', '.join(SDGQ)
            else:
                Q = st.text_area("Question:", placeholder="Enter your question here...", height=100, max_chars=5000)


    if st.button("Search!") and Q.strip()!='':
        results = general_semantic_search(Q, k=5, collection=collection)
    
        data_dict = {
            'title': [eval(str(m))['title.cleaned'] for m in results['metadatas'][0]],
            'url': [eval(str(m))['url'] for m in results['metadatas'][0]],
            'president': [eval(str(m))['president'] for m in results['metadatas'][0]],
            'json_summary': [eval(str(m))['json_summary'] for m in results['metadatas'][0]],
        }

        results_df = pd.DataFrame(data_dict)
        cols = st.columns(results_df['title'].nunique())
        unique_titles = results_df['title'].unique()

        sources = []

        related_policies = []

        for i in range(len(cols)):
            with cols[i]:
                title = unique_titles[i]
                tmp_df = results_df[results_df['title'] == title]
                source = ''
                text = ''

                for x in range(tmp_df.shape[0]):
                    source = tmp_df['url'].iloc[x].strip()
                    sources.append(source)
                    json_sum = tmp_df['json_summary'].iloc[x]
                    json_res = json.loads(json_sum)
                    keypoints_lst = list(json_res['Keypoints'].keys())
                    keypoints_desc = [json_res['Keypoints'][key]['Context'] for key in keypoints_lst]
                    keypoints_affected = [json_res['Keypoints'][key]['Policies'] for key in keypoints_lst]

                    for i in range(len(keypoints_lst)):
                        text += "... " + keypoints_lst[i] + " - " + keypoints_desc[i] + ". " + keypoints_affected[i] + '...\n\n'

                    break

                summary = generate_doc_summary(Q, text, llm)
                related_policies.append(summary)
        

        query_results(unique_titles, sources, related_policies)
    
