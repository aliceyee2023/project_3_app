# Import libraries
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_player import st_player

import pandas as pd
import numpy as np
import sklearn

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import gensim.utils

import pickle

import praw
from praw.models import MoreComments

# Set webpage name and icon
st.set_page_config(
    page_title='BiteBalance',
    page_icon='🥗',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# Font styles
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

### --- FUNCTIONS --- ###

# Define Reddit API credentials
reddit_client_id = "-rUx3v29zVVe7aMPZtnPCA"
reddit_client_secret = "rNQ7a89ilfDRLSAPEZ-3tmB9ZgwScA"
reddit_user_agent = "39 SIR Scraper"

# Initialize the Reddit API
reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent
)

# Function to scrape Reddit user's posts
def scrape_reddit_user_posts(username, num_posts=100):
    posts_dict = {
        "title": [],
        "post_text": [],
    }
    
    try:
        # Get the Reddit user instance
        user = reddit.redditor(username)

        # Iterate through the user's submissions (posts)
        for submission in user.submissions.top(limit=num_posts):
            # Append the title of each post to the list
            posts_dict['title'].append(submission.title)
            posts_dict['post_text'].append(submission.selftext)
        # Convert the dict to a dataframe
        posts_dict_df = pd.DataFrame(posts_dict)
        return posts_dict_df
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Function to clean data

def clean_data(text_df):
    # Merge title and post_text columns
    text_df['title_text'] = text_df['title'] + ' ' + text_df['post_text']

    # Remove rows with null values in the 'title_text' column
    text_df.dropna(subset=['title_text'], inplace=True)

    # Remove punctuations and tokenize using the built in cleaner in gensim
    text_df['title_text'] = text_df['title_text'].apply(lambda x: gensim.utils.simple_preprocess(x))

    # Spply stemming and stopwords exclusion within the same step
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    for idx in text_df.index:
        text_df['title_text'][idx] = [ps.stem(word) for word in text_df['title_text'][idx] if word not in stopwords]

    return text_df

# Load the trained model and vectorizer from pickle files
model_filepath = 'bernoulli_model.pkl'
vectorizer_filepath = 'tfidf_vectorizer.pkl'

with open(model_filepath, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open(vectorizer_filepath, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

# Function to display suggested content
def suggestion(subreddit_cat):
    if subreddit_cat == 'r/intermittentfasting':
        #test
        st.write('Suggested content:')
        st.write('1. Subreddit: [r/intermittentfasting](https://www.reddit.com/r/intermittentfasting/)')
        st.write('2. Gym equipment: [Colony.co](https://gymsportz.sg)')
        st.write('3. Protein supplements: [Protein la Muffins](https://www.myprotein.com.sg/nutrition/protein.list)')
        st.write('4. Grooming classes: [79 @ Anson](https://imageworks.sg/professional-image-workshop/)')
        st_player("https://youtu.be/A6Dkt7zyImk?si=Xtipaa6y0bGoeqTP")
    if subreddit_cat == 'r/AnorexiaNervosa':
        st.write('Suggested content:')
        st.write('1. Subreddit: [r/AnorexiaNervosa](https://www.reddit.com/r/AnorexiaNervosa/)')
        st.write('2. Stress Management: [40 SIB Wellness](https://www.healthhub.sg/live-healthy/be-a-master-of-stress)')
        st.write('3. Need help? [Singapore Counselling Centre](https://scc.sg/e/anorexia-nervosa/)')
        st_player("https://youtu.be/tOouAmEEnlc?si=_Zp6s89eNIVIHFa6")
    if subreddit_cat == 'Inconclusive':
        st.write('The confidence score is too low to make a conclusive prediction.')
        st.write('Please enter another user.')
        
### --- TOP NAVIGATION BAR --- ###
st.title('BiteBalance')
st.write('By 39 SIR')
selected = option_menu(
    menu_title = None,
    options = ['Insights', 'Analyse User','Analyse Text'],
    icons = ['clipboard-data','reddit','body-text'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#ff0e16'}
        }
    )

### --- 1st SECTION --- ###
if selected == 'Insights':
    # Section title
    st.header('Insights Dashboard')
    st.text("You can use the dashboard to explore the subreddits data: r/intermittentfasting and r/AnorexiaNervosa.")
    style = "<div style='background-color:#ff0e16; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # Embed dashboard
    # Define HTML content as a string
    custom_html = """
    <div class='tableauPlaceholder' id='viz1695947917091' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;pr&#47;project_3_dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='project_3_dashboard&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;pr&#47;project_3_dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1695947917091');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='777px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    # Embed HTML content into the notebook as an iframe with height that resizes to the window
    st.components.v1.html(custom_html, height=1280)

### --- 2nd SECTION --- ###
if selected == 'Analyse User':
    st.header("Reddit User Classifier")
    st.text("You can use this app to classify a Reddit user into one of two subreddits: r/intermittentfasting or r/AnorexiaNervosa.")
    style = "<div style='background-color:#ff0e16; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # Input box for Reddit user's name
    user_name = st.text_input("Enter Reddit user's name:")
    post_count = values = st.slider(
        'Select number of posts to scrape:',
        min_value=1, max_value=200, value=100, step=1)

    if st.button("Submit"):
        if not user_name:
            st.warning("Please enter a Reddit user's name.")
        else:
            st.write(f"Scraping top {post_count} posts for {user_name}")
            posts = scrape_reddit_user_posts(user_name, post_count)
            if posts is None:
                st.error("Error: User not found or unable to scrape posts.")
            else:
                st.write(f"Scrapped {len(posts.index)} posts from {user_name}.")

                # Clean the user posts using the cleaning function
                clean_data(posts)

                ### Make prediction using the pickle file

                # Vectorize the new data using the same vectorizer used during training
                X_new = loaded_vectorizer.transform(posts['title_text'].apply(lambda x: ', '.join(x)))

                # Make predictions on the new data
                predictions = loaded_model.predict(X_new)

                # Get probability estimates for the predictions
                confidence_scores = loaded_model.predict_proba(X_new)

                # Count the occurrences of each class label in the predictions
                counts = np.bincount(predictions)

                # Calculate the weighted average of confidence scores
                weighted_confidence_scores = np.zeros_like(confidence_scores[0])
                for i, prediction in enumerate(predictions):
                    weighted_confidence_scores += confidence_scores[i] * (1 / counts[prediction])

                # Define the class labels
                class_labels = ['r/AnorexiaNervosa', 'r/intermittentfasting']

                # Determine the consolidated prediction label
                consolidated_prediction_label = class_labels[np.argmax(counts)]

                # Combine the consolidated prediction label and the weighted confidence scores
                consolidated_prediction = {
                    "label": consolidated_prediction_label,
                    "confidence_scores": weighted_confidence_scores.tolist()
                }
                if weighted_confidence_scores.tolist()[0] < 0.5:
                    consolidated_prediction_label = 'Inconclusive'

                # Display the prediction
                st.write(f'Predicted subreddit: <p class="big-font">{consolidated_prediction_label}</p>', unsafe_allow_html=True)
                st.write(f"Weighted average of confidence scores: {weighted_confidence_scores.tolist()[0]}")

                # Display suggested content
                suggestion(consolidated_prediction_label)

### --- 3rd SECTION --- ###
if selected == 'Analyse Text':
    st.header("Reddit Post Classifier")
    st.text("You can use this app to classify a Reddit post into one of two subreddits: r/intermittentfasting or r/AnorexiaNervosa.")
    style = "<div style='background-color:#ff0e16; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    # Input box for Reddit post
    post_input = st.text_area("Enter a Reddit post to analyse:")
    
    if st.button("Submit"):
        if not post_input:
                st.warning("Please enter a Reddit post.")
        else:
            if post_input is None:
                st.error("Error: Please enter a Reddit post.")
            else:
                data = {'title': [post_input],
                        'post_text': [''],} 
                post_text = pd.DataFrame.from_dict(data)
                
                # Clean the post using the cleaning function
                clean_data(post_text)

                ### Make prediction using the pickle file

                with open(model_filepath, 'rb') as model_file:
                    loaded_model = pickle.load(model_file)

                with open(vectorizer_filepath, 'rb') as vectorizer_file:
                    loaded_vectorizer = pickle.load(vectorizer_file)

                # Vectorize the new data using the same vectorizer used during training
                X_new = loaded_vectorizer.transform(post_text['title_text'].apply(lambda x: ', '.join(x)))

                # Make predictions on the new data
                predictions = loaded_model.predict(X_new)

                # Get probability estimates for the predictions
                confidence_scores = loaded_model.predict_proba(X_new)
                #if confidence_scores < 0.5:
                #    subreddit = 'Inconclusive'

                if predictions == 1:
                    subreddit = 'r/intermittentfasting'
                else:
                    subreddit = 'r/AnorexiaNervosa'

                # Display the prediction
                st.write(f'Predicted subreddit: <p class="big-font">{subreddit}</p>', unsafe_allow_html=True)
                st.write(f"Confidence score: {confidence_scores[0][0]}")
                
                # Display suggested content
                suggestion(subreddit)
