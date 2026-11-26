
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import re


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# 2. CUSTOM CSS - DARK PURPLE THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #100B1D 0%,
        #1B1230 50%,
        #24163B 100%
    );
    color: #F5F0FF;
}

/* Main Content */
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
}

/* Main Title */
.main-title {
    text-align: center;
    color: #D8BFFF;
    font-size: 36px;
    font-weight: bold;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #B9A8D5;
    font-size: 17px;
    margin-bottom: 25px;
}

/* All Headings */
h1, h2, h3 {
    color: #E9DDFF !important;
}

/* Normal Text */
p, label, .stMarkdown, .stCaption {
    color: #E5DDF2;
}

/* Text Area */
.stTextArea textarea {
    background-color: #211832 !important;
    color: #FFFFFF !important;
    border: 1px solid #7250A5 !important;
    border-radius: 12px !important;
}

.stTextArea textarea:focus {
    border: 1px solid #B58AFF !important;
    box-shadow: 0 0 0 1px #B58AFF !important;
}

.stTextArea textarea::placeholder {
    color: #A99ABF !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        90deg,
        #7041A6,
        #9562D5
    );
    color: #FFFFFF !important;
    border: 1px solid #A77BE0;
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #8956C5,
        #AD83E9
    );
    border-color: #D1B3FF;
    color: #FFFFFF !important;
}

/* Positive Result Card */
.result-card {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 18px;
    border-radius: 14px;
    margin: 15px 0;
    font-family: sans-serif;
}

.positive {
    background: #102D27;
    border: 1.5px solid #00D68F;
    color: #FFFFFF;
}

/* Negative Result Card */
.negative {
    background: #351B2A;
    border: 1.5px solid #FF526F;
    color: #FFFFFF;
}

/* Result Symbols */
.symbol {
    font-size: 42px;
    font-weight: bold;
    width: 65px;
    text-align: center;
}

/* Result Details */
.details {
    flex: 1;
    min-width: 0;
}

.label {
    font-size: 15px;
    color: #D8CDE8;
}

.sentiment {
    font-size: 25px;
    font-weight: bold;
}

.positive-text {
    color: #00E69A;
}

.negative-text {
    color: #FF647E;
}

/* Prediction Score */
.score {
    font-size: 27px;
    font-weight: bold;
    color: #FFFFFF;
}

/* Progress Bar */
.bar {
    height: 12px;
    background: #302541;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 8px;
}

.positive-fill {
    height: 100%;
    background: #00E69A;
    border-radius: 10px;
}

.negative-fill {
    height: 100%;
    background: #FF526F;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #49375F !important;
}

/* Alerts */
.stAlert {
    background-color: #302344;
    color: #FFFFFF;
    border: 1px solid #805BB5;
}

/* Responsive Design */
@media (max-width: 600px) {

    .result-card {
        gap: 10px;
        padding: 12px;
    }

    .symbol {
        width: 35px;
        font-size: 28px;
    }

    .sentiment {
        font-size: 19px;
    }

    .score {
        font-size: 20px;
    }

    .main-title {
        font-size: 27px;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🎬 IMDB Movie Review Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze movie reviews using Deep Learning '
    'and discover their sentiment.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. LOAD IMDB WORD INDEX
# --------------------------------------------------

@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


word_index = load_word_index()


# --------------------------------------------------
# 5. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    return load_model("simple_rnn_imdb.h5")


model = load_sentiment_model()


# --------------------------------------------------
# 6. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):

    words = re.findall(r"\b\w+\b", text.lower())

    encoded_review = []

    for word in words:

        index = word_index.get(word)

        if index is None or index + 3 >= 10000:
            encoded_review.append(2)
        else:
            encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=300
    )

    return padded_review


# --------------------------------------------------
# 7. SENTIMENT PREDICTION
# --------------------------------------------------

def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)

    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )[0][0]

    positive_score = float(prediction)

    negative_score = 1 - positive_score

    sentiment = (
        "Positive" if positive_score > 0.5
        else "Negative"
    )

    return sentiment, positive_score, negative_score


# --------------------------------------------------
# 8. SAMPLE REVIEWS
# --------------------------------------------------

def set_positive_review():

    st.session_state.review_input = (
        "This movie was absolutely amazing! "
        "The acting was excellent and the story "
        "was fantastic. I really enjoyed it."
    )


def set_negative_review():

    st.session_state.review_input = (
        "This movie was terrible. The story was boring, "
        "the acting was very poor and I did not enjoy it."
    )


def clear_review():

    st.session_state.review_input = ""


# --------------------------------------------------
# 9. SAMPLE REVIEW BUTTONS
# --------------------------------------------------

st.subheader("💡 Try a Sample Review")

col1, col2 = st.columns(2)

with col1:

    st.button(
        "👍 Positive Review",
        use_container_width=True,
        on_click=set_positive_review
    )

with col2:

    st.button(
        "👎 Negative Review",
        use_container_width=True,
        on_click=set_negative_review
    )


# --------------------------------------------------
# 10. MOVIE REVIEW INPUT
# --------------------------------------------------

st.subheader("💬 Enter Your Movie Review")

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

user_input = st.text_area(
    "Movie Review",
    placeholder="Write your movie review here...",
    height=150,
    key="review_input"
)


# Character Counter

st.caption(
    f"📝 Characters: {len(user_input)}"
)


# Clear Review Button

st.button(
    "🗑️ Clear Review",
    on_click=clear_review
)


# --------------------------------------------------
# 11. ANALYZE SENTIMENT BUTTON
# --------------------------------------------------

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if user_input.strip():

        sentiment, positive_score, negative_score = (
            predict_sentiment(user_input)
        )

        st.session_state.prediction = (
            sentiment,
            positive_score,
            negative_score
        )

    else:

        st.warning("Please enter a movie review.")


# --------------------------------------------------
# 12. DISPLAY PREDICTION RESULTS
# --------------------------------------------------

if "prediction" in st.session_state:

    sentiment, positive_score, negative_score = (
        st.session_state.prediction
    )

    st.subheader("📊 Prediction Result")


    # Positive Result Card

    st.html(f"""
    <div class="result-card positive">

        <div class="symbol" style="color:#00E69A;">
            ✓
        </div>

        <div class="details">
            <div class="label">Sentiment</div>
            <div class="sentiment positive-text">
                Positive
            </div>
        </div>

        <div class="details">
            <div class="label">Prediction Score</div>

            <div class="score">
                {positive_score * 100:.2f}%
            </div>

            <div class="bar">
                <div class="positive-fill"
                style="width:{positive_score * 100:.2f}%;"></div>
            </div>
        </div>

    </div>
    """)


    # Negative Result Card

    st.html(f"""
    <div class="result-card negative">

        <div class="symbol" style="color:#FF647E;">
            ✕
        </div>

        <div class="details">
            <div class="label">Sentiment</div>
            <div class="sentiment negative-text">
                Negative
            </div>
        </div>

        <div class="details">
            <div class="label">Prediction Score</div>

            <div class="score">
                {negative_score * 100:.2f}%
            </div>

            <div class="bar">
                <div class="negative-fill"
                style="width:{negative_score * 100:.2f}%;"></div>
            </div>
        </div>

    </div>
    """)


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<center>Built with ❤️ using Streamlit & Deep Learning</center>",
    unsafe_allow_html=True
)
