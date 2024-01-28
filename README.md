# RUGPULL

## What is RUGPULL?

RUGPULL is a simple python script paired with a Javascript GUI that lets you perform RAG on a single document. It is in very early stages of development - features coming later will be the ability to perform RAG on multiple documents at once, better visualization/UI, and more customization. 

RUGPULL is USEFUL for:
- Seeing what parts of a document are relevant to a query
- Visualizing the process of RAG
- Searching through large documents efficiently (the spatial features of rugpull help here)

## Setup

### Python
To use RUGPULL, you will need to install the following python packages:
```
pip install -r requirements.txt
```
Preferably in a virtual environment.

Then, just `python main.py` and navigate to `localhost:3000` in your browser - you can interact with RUGPULL from there.


## Rugpull is a contrived acronym for:
### RUGPULL
### Retrieval
### yoU
### Generate
### Plus
### yoU
### Leverage
### Learning

## How do I use RUGPULL?
RUGPULL includes a variety of tools for you to specifically search through documents yourself - it embeds sentence fragments into a vector space, and then allows you to search for similar sentences to the fragments you provide on an interactive map,
where individual contexts are represented as points - colored with respect to their relevance to your query.

You can also just see the results of a simple cosine similarity search, which will return the most similar contexts to your query. You can also press the search button, provided you have LM Studio open and running its local server, which will then call whatever LLM you have loaded in LM Studio to generate a proper response to your query that cites the most relevant contexts. Good LLMs for this task include openhermes-2.5, dolphin-2.7-mixtral (if your computer has the RAM), or really any modern mistral-based instruction tuned model.

Rugpull *requires no internet* to run (except when downloading models), and is entirely local. It is also entirely open source, and you can see exactly what it is doing at any time.


## Hardware requirements

RUGPULL is a heavy application, and the primary bottleneck will be processing local embeddings for exceedingly long documents - however, a standard novel can be processed in a few minutes on a modern CPU. The GUI is also quite heavy, and will require a decent amount of RAM to run smoothly. However, the search function is highly optimized and can search through 100 million rows in 5 milliseconds on a modern GPU.

## Embeddings

RUGPULL stores embeddings in .npy files in subdirectories of a 'cache' folder, initialized in the directory RUGPULL is in. This is to avoid having to recompute embeddings for documents you have already processed. If you want to clear the cache, just delete the cache folder. These subdirectories can be passed around, so the heavy lifting of embedding a document can be done on a more powerful machine, and then the cache can be passed to a less powerful machine to be searched through.

## Support/Advice

RUGPULL is currently in pre-alpha. Open github issues, or DM on twitter @N8Programs (or my discord, n8programs) for support. I'm happy to help with any issues you may have, or any advice you may need.

Active development is ongoing. Expect things to change, fast.