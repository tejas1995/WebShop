from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
import json
import pdb

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.engine.goal import get_human_goals, get_synthetic_goals
from tejas_code.openai_utils import openai_caller


NUM_TASKS = 100
MAX_STEPS = 30

#env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=1000)
env = WebAgentTextEnv(observation_mode='text_rich', num_products=1000, human_goals=False)
s = env.browser.server
all_products = s.all_products
product_prices = s.product_prices

goals = get_synthetic_goals(all_products, product_prices)

gpt_initial_sys_prompt = """You are an agent who is trying to help a user find an item on a shopping website. 
The observation contains information about what you can see on the current webpage. 
You can navigate the website using two types of actions:
1. search actions: search items on the website by passing in a query (preferably concise)
2. click actions: these can help you navigate the website (`back`, `next >`, `prev >`), explore items (by product code), learn about features and read reviews, and buy items that fit the user's criteria.
You will be told which buttons you can click on the current webpage, and whether search is available or not.
You will also be shown the history of observations you have already seen and actions you have already taken.
Your job is to satisfy the user's goal by taking actions in the website environment, namely making search queries, navigating the website by clicking on webpages to explore candidate products, and executing items such as buying.


Here's an example sequence of observations and actions:"""

gpt_action_prompt = """Choose the next action (either a search, if it is available, or click option). 
The action must have exactly one argument. In the case of search, this will be a search query (under 5 words), and in case of click, it will be one of the options from the clickable options list provided. 
As a reminder, here are some properties of the actions that you can take:
1. Search actions: executing this will give you a list of products, and for each product you will see the product code, name and price. This is only partial information about each product, if you need more information to see if the product matches the user's criteria you will need to click on the product button. Do not repeat the search query you have already seen in the history. Do not include information about the product's options (such as size, color, etc.) or price in the search query, since those will be visible only when you open the product page.
2. Click actions: these can help you navigate the website (`back`, `next >`, `prev >`), explore items that may fit the user's needs (by product code), learn about features and read reviews, and buy items that fit the user's criteria. If you see information about a product that fits some but not all criteria, you should click on it to learn more about it.

You can only take actions that are available on the current page (i.e. the last observation). For example, if you go to the features page of a product and want to buy the product, you will need to go back to the previous page where the "Buy Now" button is available before you can buy it.
If you have not found a relevant product beyond the first few pages of search results, you should consider changing your search query to something more relevant.

Enclose your action argument in []. Examples of valid actions are `search[shoes]`, `click[next >]` and `click[PRODUCT_CODE]`.
Think step-by-step about what is the best action to take (using fewer than 100 tokens), and then output your final action within <action></action> tags.
"""

example = [
    {"role": "user", "content": """Instruction:    
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]    

"""},
    {"role": "assistant", "content": """I should begin by making a relevant search query.\n\n<action>search[3 ounce bright citrus deodorant sensitive skin]</action>"""},
    {"role": "user", "content": """[Back to Search] 
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

"""},
    {"role": "assistant", "content": """The first result might fit the bill, let me click it to explore more.<action>click[B078GWRC1J]</action>"""},
    {"role": "user", "content": """[Back to Search] 
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

"""},
    {"role": "assistant", "content": "For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.\n\n<action>click[bright citrus]</action>"},
    {"role": "user", "content": """You have clicked bright citrus. 

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

"""},
    {"role": "assistant", "content": "The item is a 3 ounce bottle of bright citrus deodorant for sensitive skin, and the price is $10.99, which is lower than $50.00. Let me select the correct size option as well.\n\n<action>click[3 ounce (pack of 1)]</action>"},
    {"role": "user", "content": """You have clicked 3 ounce (pack of 1). 

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

"""},
    {"role": "assistant", "content": "Looks like I have satisfied all the requirements.\n\n<action>click[Buy Now]</action>"},
]

total_reward = 0
orig_cost = openai_caller.compute_cost()
all_trajectories = []
for g in tqdm(range(NUM_TASKS)):
    # env.reset(env.browser.session_id)
    env.reset(g)
    interaction_history = []

    session = env.browser.server.user_sessions[env.browser.session_id]
    goal = session['goal']

    print("USER:", env.instruction_text)
    observation = env.observation
    for i in range(MAX_STEPS):
        orig_observation = env.observation
        # print("OBSERVATION:", observation)
        available_actions = env.get_available_actions()

        prompts = [{
            "role": "system", 
            "content": gpt_initial_sys_prompt
        }]
        prompts += example
        prompts += [{
            "role": "system", 
            "content": f"Now execute actions to solve the following instruction.\n\nUser instruction: {env.instruction_text.replace('Instruction: ', '')}"
        }]
        for item in interaction_history:
            prompts.append({"role": "user", "content": f"Observation: {item['observation']}"})
            prompts.append({"role": "assistant", "content": f"Agent Action: {item['action']}"})
        final_prompt = f"Current observation: {observation}\n\n"
        final_prompt += "Search option is " + "" if available_actions['has_search_bar'] else "not " + "available.\n"
        final_prompt += "Clickable options: " + str(available_actions['clickables']) + "\n\n"
        # final_prompt += gpt_action_prompt
        prompts.append({"role": "user", "content": final_prompt})
        prompts.append({"role": "system", "content": gpt_action_prompt})
        gpt_output = openai_caller(prompt=prompts, model='gpt4o', max_new_tokens=150, temperature=0.0)
        # print(prompts[-1])
        action = gpt_output.split("<action>")[-1].split("</action>")[0]
        # print("AGENT:", gpt_output.replace("[", "(").replace("]", ")"))
        # print("-"*100)
        observation, reward, done, info = env.step(action)
        interaction_history.append({
            "observation": orig_observation,
            "action": gpt_output
        })
        if done:
            # print("DONE")
            # print(reward)
            interaction_history.append({
                "observation": observation,
            })
            break
    total_reward += reward
    trajectory = {
        "goal_idx": g, 
        "goal": goal,
        "interaction_history": interaction_history,
        "reward": reward,
        "completed": done,
        "num_steps": len(interaction_history),
    }
    all_trajectories.append(trajectory)

print("Total tasks completed:", NUM_TASKS)
print(f"Total cost: ${openai_caller.compute_cost() - orig_cost:.2f}")
print(f"Total reward: {total_reward}")
json.dump(all_trajectories, open(f"tejas_code/gpt4o_react_eval-{NUM_TASKS}tasks-{MAX_STEPS}maxsteps.json", "w"), indent=2)
pdb.set_trace()