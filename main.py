from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
    summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

################################## UI ##############################################

def st_ui():
    my_text = st.text_area("Text to summarize", open("./resources/example_text.txt", "r").read())
    if st.button("Generate"):
        with st.spinner('Generating summary...'):
            summary = generate_summary(my_text)
            st.write(summary)
    
if __name__ == "__main__":
    st_ui()

