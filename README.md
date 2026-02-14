# Intro

To use this program with ease, go to the streamlit website listed below

i used a google api key to configure this project since it can be generated for free and it has 25 free uses with the code 

to use this program though, a minimum of 4 agents will be invoked, meaning you can have around 6 tries per day of this program, if the agents are all only called once


# generating a key and using the website:

to generate a key to use it in the website, go to https://aistudio.google.com/ and create an account

then generate an api key through their website website by going to the dashboard, clicking create api key at the top right, and creating that key

the key will be a really long name that you will need to paste into the website

i should also note that when you enter the key, it is stored locally on your ram in the browser tab, not on the website or anything adjacent to the website



# streamlit website url (to use the app): 

https://agenticcodingpipeline-9ko4n9cu8ek4fw42dxsseg.streamlit.app/

# local use

It is also possible to use this program locally by downloading the file, installing the libraries outlined in requirements.txt, and inputting your api key in a separate .env file but i would recommend using the website for ease of use.

also, if you were going to use it locally, it isn't that safe to use the sys.executable line in a non-containerized environment so i would comment that out and comment in the docker command in the python library so that you can use a containerized environment. this would also mean you have docker desktop installed and open to use it.
