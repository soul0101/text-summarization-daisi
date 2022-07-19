from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
# tokenizer = AutoTokenizer.from_pretrained("./model/")
# model = AutoModelForSeq2SeqLM.from_pretrained("./model/")

# text ="""
# The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
# The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
# At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
# "We'll be the comeback kids, all of us," he said. "We want to get our country back."
# The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
# """

# inputs = tokenizer.encode("summarize: " + text,
#         return_tensors='pt',
#         max_length=512,
#         truncation=True
#     )
# summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
# summary = tokenizer.decode(summary_ids[0])
# print(summary)

def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text,
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
    summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
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

