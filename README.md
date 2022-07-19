# text-summarization-daisi
Document summarization uses natural language processing techniques to generate a summary for documents. There are two general approaches to auto-summarization: extractive and abstractive. The technique for document summarization is extractive.

This feature extracts sentences that collectively represent the most important or relevant information within the original content. It locates key sentences in an unstructured text document, and collectively, these sentences convey the main idea of the document. <br>

## Use Cases:
1) Distill critical information from lengthy documents, reports, and other text forms
2) Highlight key sentences in documents
3) Generate news feed content

## Test API Call
```python
import pydaisi as pyd

text_summarizer = pyd.Daisi("soul0101/Text Summarizer")

summary = text_summarizer.generate_summary(
    """
    Daisi brings the power of cloud computing into the hands of every developer, scientist, engineer by automatically deploying and creating endpoints for any function of a Python code, which can then be invoked seamlessly from any environment.
    Any regular Python code can be turned into a Daisi by simply registering its Github repository in the Daisi platform, with no need to write any specific code.
    Here's a video overview of our Daisi Hackathon with US $10,000 in Prizes!
    """, min_length=30).value

print(summary)
```
## Reference:
Summarization model used: https://huggingface.co/sshleifer/distilbart-cnn-12-6
