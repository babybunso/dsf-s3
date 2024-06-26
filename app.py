import streamlit as st
import pandas as pd


#####     CONSTANTS     #####
SDG = ["No Poverty", "Zero Hunger", "Good Health and Well-being", "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy", "Decent Work and Economic Growth", "Industry, Innovation and Infrastructure", "Reduced Inequalities", "Sustainable Cities and Economies", "Responsible Consumption and Production", "Climate Action", "Life Below Water", "Life on Land", "Peace, Justice and Strong Institutions", "Partnership for the Goals"]
APP_NAME = 'SONAS: Semantic OpenAI NLP Assistant System'
APP_DESC = ' `@Team Rods` - Medyo mahiyain pa, pero sakto lang ang Group Dynamix'
COLOR_BLUE = '#0038a8'
COLOR_RED = '#ce1126'
COLOR_GRAY = '#f8f8f8'


#####     FUNCTIONS     #####


##### STYLED COMPONENTS #####
# set universal styles
html_styles = f"""
    <style>
        h3 {{
            color: {COLOR_BLUE};
        }}

        p {{
            font-size: 1.125rem;
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
    </style>
"""

def keywords_line(keywords_lst):
    """Function to style the keywords list (below the title)"""
    # st.markdown("<br>", unsafe_allow_html=True)

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


def summary_cards(eng_sum, fil_sum):
    """Function to style the summary columns"""
    # st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap='medium')

    col1_design = f"""
    <style>
        div.left{{
            border: 1px, solid, {COLOR_GRAY};
            border-radius: 20px;
            background-color: {COLOR_GRAY};
            padding: 15px 30px;
        }}
    </style>

    <div class="left">
        <h3><span class='bolded'>English</span> Summary</h3>
        <p>{eng_sum}</p>
    </div>
    """

    col2_design = f"""
    <style>
        div.right {{
            padding: 15px 30px;
        }}
    </style>

    <div class="right">
        <h3><span class='bolded'>Filipino</span> Summary</h3>
        <p>{fil_sum}</p>
    </div>
    """

    with col1:
        st.markdown(col1_design, unsafe_allow_html=True)
    with col2:
        st.markdown(col2_design, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def actions_layout(point_lst, desc_lst, impact_lst, presi, is_eng):
    """Function to style the action + implication section"""
    #st.markdown("<br>", unsafe_allow_html=True)
    imp_header = ""

    if is_eng:
        st.subheader("Policies and Plans")
        st.markdown(f"<p class=italicized>The following were the key policies and plans mentioned by president {presi}</p>", unsafe_allow_html=True)
        imp_header = "Potential Impacts"
    else:
        st.subheader("Mga Plano at Patakaran")
        st.markdown(f"<p class=italicized>Ang mga sumusunod ay ang mga patakaran at plano na binanggit ni pangulong {presi}</p>", unsafe_allow_html=True)
        imp_header = "Potential Impacts but in Fil"

    for i in range(len(point_lst)):
        actions_html = f"""
        <style>

        </style>
        <h4>{point_lst[i]}</h4>
        <p>{desc_lst[i]}</p>
        """

        impacts_html = f"""
        <p>{impact_lst[i]}</p>
        """

        st.markdown(actions_html, unsafe_allow_html=True)

        with st.expander(imp_header, expanded=False):
            st.markdown(impacts_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


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
        <p class="bolded info-header">Primary Theme of SONAs:</p>
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
# Load SONA dataset
df = pd.read_csv("data/sonas.csv")

# Create streamlit app
st.set_page_config(layout='wide')
st.title(APP_NAME)
st.write(APP_DESC)
st.markdown(html_styles, unsafe_allow_html=True)

options = st.sidebar.radio("", ["Home", "SONA Summarizer", "Presidential Analysis", "Topic Analysis"])

##################### TO REMOVE #####################
eng_sum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum condimentum, ante vel interdum imperdiet, enim diam euismod dolor, hendrerit imperdiet orci ipsum et dolor. Nullam congue id libero sed ultrices. Mauris facilisis convallis massa, ac ornare leo fermentum ac. Quisque ultricies ultrices orci, consequat feugiat turpis cursus tempus. Pellentesque facilisis, mauris quis tincidunt consequat, urna mi aliquam velit, gravida euismod dui libero in dolor. Sed sed elit eu sem fringilla interdum eu vel urna. Duis sollicitudin fermentum fringilla. Nunc rhoncus, magna at finibus accumsan, urna risus facilisis est, pretium dignissim tortor dolor quis ex. Integer viverra malesuada condimentum. Sed at quam congue, varius felis eget, venenatis risus. Aenean ultrices orci sit amet ex pharetra tempor. Ut a aliquam mi, id accumsan magna. Mauris a blandit leo. Sed et erat neque."

fil_sum = "Praesent egestas tristique justo, a tempor odio gravida non. Curabitur diam nisl, egestas vitae fermentum ac, pellentesque at risus. Ut sagittis lectus ac mollis auctor. Suspendisse nec urna nec nisi auctor commodo non ac nibh. In vitae est nec leo scelerisque pulvinar sit amet ac felis. Etiam porta euismod laoreet. Duis ullamcorper, justo in placerat fringilla, enim ante consectetur arcu, sit amet ultricies ligula dui iaculis neque. Nullam eu augue aliquam, convallis tellus eget, fringilla velit. Ut ac maximus nunc, vel molestie magna. Ut non magna nisi. Vestibulum cursus urna varius eros mattis, ut accumsan ante consectetur. Donec quis leo pretium, sagittis nibh non, maximus libero. Proin rhoncus tortor ut tellus elementum, porttitor tristique velit congue. Class aptent taciti sociosqu."

theme = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum condimentum, ante vel interdum imperdiet, enim diam euismod dolor, hendrerit imperdiet orci ipsum et dolor. Nullam congue id libero sed ultrices. Mauris facilisis convallis massa, ac ornare leo fermentum ac. Quisque ultricies ultrices orci, consequat feugiat turpis cursus tempus. Pellentesque facilisis, mauris quis tincidunt consequat, urna mi aliquam velit, gravida euismod dui libero in dolor. Sed sed elit eu sem fringilla interdum eu vel urna. Duis sollicitudin fermentum fringilla."

tempt_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque  blandit, ligula vel venenatis elementum, elit orci cursus ante, sit amet  ultricies dui risus in nisi. Aenean sed vehicula quam, non varius est.  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut mi risus,  elementum at orci vitae, cursus aliquam nisi. Aenean ante nunc, rhoncus  ut pellentesque eget, varius vitae augue. Quisque sapien nibh, dapibus  non ligula sed, semper pulvinar lorem. Donec mi neque, lacinia vel diam  ac, egestas vestibulum sapien. Donec semper faucibus nunc, ultrices  dapibus justo fringilla sit amet. Vestibulum tempus massa eget mauris  fringilla consectetur. Sed venenatis commodo."

tempt_text_fil = "IN FILIPINO " + tempt_text
##################### TO REMOVE #####################

if options == "Home":
    st.write('___')
    st.write("teka lang")


elif options == "SONA Summarizer":
    st.write('___')
    title = st.selectbox('Choose a SONA to Summarize', df['title.cleaned'])
    if title:
        sona_df = df[df['title.cleaned'] == title]
        presi = sona_df['president'].iloc[0]
        speech = sona_df['speech'].iloc[0]
        link = sona_df['url'].iloc[0]
        keywords = ["test 1", "test 2"]

        st.header(title)
        keywords_line(keywords)
        st.caption(link)

        summary_cards(eng_sum, fil_sum)
        
        filipino_ver_toggle = st.toggle(label="Use Filipino Translation", value=False)

        if filipino_ver_toggle:
            # put filipino version of the outputs
            point_lst = ['1. Blah blah blah (Fil ed.)', '2. Smth smth (Fil ed.)']
            desc_lst = [tempt_text_fil, tempt_text_fil]
            impact_lst = [tempt_text_fil, tempt_text_fil]
            actions_layout(point_lst, desc_lst, impact_lst, presi, False)
            pass

        else:
            # put english version of the outputs
            point_lst = ['1. Blah blah blah', '2. Smth smth']
            desc_lst = [tempt_text, tempt_text]
            impact_lst = [tempt_text, tempt_text]
            actions_layout(point_lst, desc_lst, impact_lst, presi, True)
        
        


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

        president_card(presi, img_link, info, theme)


elif options == "Topic Analysis":
    st.write('___')
    with st.expander("Query Search", expanded=True):
            query_type = st.radio("Select a Query Type", ['SDG', 'Text Area'])
            if query_type == "SDG":
                title = st.selectbox('SDG Goals', SDG)
            else:
                Q = st.text_area("Ask Question:", placeholder="Enter your question here...", height=100, max_chars=5000)
            st.button("Search!")
    
    
