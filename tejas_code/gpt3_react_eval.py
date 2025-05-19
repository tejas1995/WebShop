from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
import json
import pdb
import sys
import os

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.engine.goal import get_human_goals, get_synthetic_goals

import openai
openai.api_key = os.environ['OPENAI_API_KEY']


NUM_TASKS = 100
MAX_STEPS = 15

#env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
env = WebAgentTextEnv(observation_mode='text_rich', num_products=1000, human_goals=False)

def llm(prompt, stop=["\n"]):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response["choices"][0]["text"]

# trivial search & item, choose option
prompt1 = """Webshop 
Instruction:    
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]    

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95    

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]    

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

def webshop_run(idx, prompt, to_print=True):
    action = 'reset'
    init_prompt = prompt
    prompt = ''
    for i in range(MAX_STEPS):

        if action.startswith('think'):
            observation = 'OK.'
        elif action == 'reset':
            env.reset(idx)
            observation = env.observation
            res = (observation, None, False, None)
        else:
            try:
                res = env.step(action)
                observation = res[0]
            except AssertionError:
                observation = 'Invalid action!'


        if to_print:
            print(f'Action: {action}')#\nObservation: {observation}\n')
            sys.stdout.flush()
        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
        else:
            prompt += f'{observation}\n\nAction:'
        
        if res[2]:    
            return res[1]

        action = llm(init_prompt + prompt[-(6400-len(init_prompt)):], stop=['\n']).lstrip(' ')

    return 0

def run_episodes(prompt, n=50):
    rs = []
    cnt = 0
    for i in tqdm(range(n)):
        print('-----------------')
        print(i)
        try:
            r = webshop_run(f'fixed_{i}', prompt, to_print=True)
        except AssertionError:
            r = 0
            cnt += 1
        rs.append(r)
        if (i+1) % 1 == 0:
            r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
            print(i+1, r, sr, fr)
            print('-------------')
    r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
    print(r, sr, fr)
    return rs

if __name__ == '__main__':
    run_episodes(prompt1, n=1)