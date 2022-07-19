from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
# model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
tokenizer = AutoTokenizer.from_pretrained("./model/")
model = AutoModelForSeq2SeqLM.from_pretrained("./model/")

@st.cache(suppress_st_warning=True)
def generate_summary(text, min_length = 80, max_length=150):
    """
    Generate a summarized version of given text.
    
    Parameters
    ----------
    min_length: int
        The minimum word count of the summary
    max_length: int
        The maximum word count of the summary  

    Returns
    -------
    str
    """
    inputs = tokenizer.encode("summarize: " + text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@st.cache(suppress_st_warning=True)
def get_word_count(text):
    """
    Get word count of a string
    
    Parameters
    ----------
    text: str

    Returns
    -------
    int
    """
    return len(text.split())

################################## UI ##############################################

def st_ui():
    st.write("# Welcome to the Text Summarizer Daisi! 👋")
    st.markdown(
        """
            This daisi allows you to obtain a summarized and concise version of a long text.
        """
    )
    col1, col2 = st.columns([1,1])
    with col1:
        st.title("Text")
        my_text = st.text_area("", open("./resources/example_text.txt", "r").read(), height=300, key='text_key')
        word_count = get_word_count(my_text)
        st.write("Word Count: %s" % (word_count))
    with col2:
        st.title("Summarized Text")
        sum_text = st.empty()
        sum_text.text_area("", "", height=300, disabled=True)
        values = st.slider(
            'Select a range for summary word count',
            0, word_count, (min(30, word_count), min(150, word_count)))

    generate_btn = st.button('Generate')
    if generate_btn:
        sum_text.text_area("", "Generating...", height=300, disabled=True)
        summary = generate_summary(my_text, max_length=values[1], min_length=values[0])
        sum_text.text_area("", summary, height=300)
                

    
if __name__ == "__main__":
    st_ui()

