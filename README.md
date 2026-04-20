# Initial Setup

The best method of using this application would be to fork the Agentic Coding Pipeline repository and download the files onto your machine locally. 

First, in order to begin to use this project, make sure to have python installed on your machine.

After you have forked the project or downloaded the files locally and you are located in the project folder for this application, you need to use the command “pip install -r requirements.txt” to install all of the dependencies for this project. 

In order to use the research agent for this project, you will also have to download and unzip a chroma_db file in the folder that the project is stored in because this file has the python documentation information. The link to this zip file can be found here: https://drive.google.com/drive/folders/1RssBurqRLTgIKLFLsrASnjntJknBDtys?usp=sharing

# Running the Project

After all of the dependencies are installed and you have unzipped the chroma_db folder, you can now run the application with “streamlit run agents.py”.

Usually, the website will pop up after this command starts or you can go to the local host or network host links that appear in the terminal after you run this command.

After the page loads, there will be a configuration sidebar in order to use the page. 

You must enter in the provider, api key, and model id and then press ‘submit config’ to access the rest of the application.

# Generating a key for the configuration:

I will show the instructions for generating a google api key and have free tries. Feel free to use any other provider such as claude, openrouter, ollama.

to generate a key to use it in the website, go to https://aistudio.google.com/ and create an account or go to any ap

then generate an api key through their website website by going to the dashboard, clicking create api key at the top right, and creating that key

the key will be a really long name that you will need to paste into the website

# After configuration

After the configuration is submitted, you now have access to the website. Simply enter in a prompt and press ‘Run {mode} Orchestration Loop’ to start the loop. 

An optional checkbox is presented if you want to run a research agent that will gather details from python documentation, web information, or both before running the complete loop. 

Switch tabs to switch modes.

The final tab is designated so a user can view the benchmark dashboard.

# FAQs

Q: Why does the “pip install -r requirements.txt” or “streamlit run agents.py” commands not work
A: Make sure pip can interact with the terminal or in the system path. Also, running “python -m streamlit run agents.py” may be a solution for the streamlit error.

Q: How do I get an API key?
A: There are multiple providers listed with their own instructions on how to get API keys. Every service is different and the application does not provide one since it would cost a lot of money.
