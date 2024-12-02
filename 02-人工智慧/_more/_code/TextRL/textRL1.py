from textrl import TextRLEnv,TextRLActor
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
import logging
import sys
import pfrl
import torch
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

tokenizer = AutoTokenizer.from_pretrained("huggingtweets/elonmusk")  
model = AutoModelWithLMHead.from_pretrained("huggingtweets/elonmusk")
model.eval()
model.cuda()
sentiment = pipeline('sentiment-analysis',model="cardiffnlp/twitter-roberta-base-sentiment",tokenizer="cardiffnlp/twitter-roberta-base-sentiment",device=0,return_all_scores=True)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.CRITICAL)

r = sentiment("dogecoin is bad")[0][0]['score']
print('sentiment("dogecoin is bad")[0][0][score]=', r)

class MyRLEnv(TextRLEnv):
    def get_reward(self, input_item, predicted_list, finish): # predicted will be the list of predicted token
      reward = 0
      if finish or len(predicted_list) >= self.env_max_length:
        if 1 < len(predicted_list):
          predicted_text = tokenizer.convert_tokens_to_string(predicted_list)
          # sentiment classifier
          reward += sentiment(input_item[0]+predicted_text)[0][0]['score']
      return reward

observaton_list = [['i think dogecoin is']]

env = MyRLEnv(model, tokenizer, observation_input=observaton_list)
actor = TextRLActor(env,model,tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=10, epochs=10)

print('observation_list[0]=', observaton_list[0])
r = actor.predict(observaton_list[0])
print('predict=', r)

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=100,
    eval_n_steps=None,
    eval_n_episodes=1,       
    train_max_episode_len=100,  
    eval_interval=10,
    outdir='elon_musk_dogecoin', 
)

agent.load("./elon_musk_dogecoin/best")

r = actor.predict(observaton_list[0])
print('predict=', r)
