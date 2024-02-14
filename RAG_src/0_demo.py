import os


'''
from embedchain import Pipeline as App
# api_key = os.environ.get("OPENAI_API_KEY")
# if api_key is None:
#     raise ValueError("OPENAI_API_KEY is not set in the environment variables")


# Create a bot instance

elon_bot = App()

# Embed online resources
elon_bot.add("https://en.wikipedia.org/wiki/Elon_Musk")
elon_bot.add("https://www.forbes.com/profile/elon-musk")

# Query the bot
elon_bot.query("How many companies does Elon Musk run and name those?")
# Answer: Elon Musk currently runs several companies. As of my knowledge, he is the CEO and lead designer of SpaceX, the CEO and product architect of Tesla, Inc., the CEO and founder of Neuralink, and the CEO and founder of The Boring Company. However, please note that this information may change over time, so it's always good to verify the latest updates.
'''


from embedchain import App

os.environ["OPENAI_API_KEY"] = "sk-"

config_dict = {
  'llm': {
    'provider': 'openai',
    'config': {
      'model': 'gpt-3.5-turbo',
      'temperature': 0.0,
      'max_tokens': 20,
      'top_p': 1,
      'stream': False
    }
  },
  'embedder': {
    'provider': 'openai'
  }
}

# load llm configuration from config dict
app = App.from_config(config=config_dict)



# json test
# app = App()

# Add json file
# app.add("temp.json")
# with open("v5.freetext.t50.txt") as f:
with open("v5.freetext.500.txt") as f:
    content = f.read()
app.add(content)
import pdb;pdb.set_trace()
res = app.query("What is the definition of SVM?")
print (res)
'''
head 500:
Support Vector Machines (SVM) is a machine learning algorithm used for classification and regression analysis. It is a supervised learning model that analyzes data and separates it into different classes or categories. SVM works by finding the optimal hyperplane that maximally separates the data points of different classes.

tail 50:
I don't know the answer to the query.

'''