"""
Some code is taken from ACDC paper: https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc
"""

from transformers import AutoTokenizer
import random
import json



NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", 
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance", 
    "rule", "sanctions", "shift", "siege", "slump", 
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour", 
    "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work",
]

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

YEARS = []
YEARS_BY_CENTRURY = {}
for century in range(11, 18):
    all_success = []

    year01 = century * 100 + 1
    assert tokenizer.encode(f" {year01}") == [tokenizer.encode(f" {str(year01)[:2]}")[0], tokenizer.encode(str(year01)[2:])[0]], year01

    for year in range(century * 100 + 2, (century * 100) + 100):
        a = tokenizer.encode(f" {year}")
        if a == [tokenizer.encode(f" {str(year)[:2]}")[0], tokenizer.encode(str(year)[2:])[0]]:
            all_success.append(str(year))
            continue
    YEARS.extend(all_success[:-1])
    YEARS_BY_CENTRURY[str(century)] = all_success


def get_prompts(num_examples, seed):
    random.seed(seed)
    prompts = []
    for i in range(num_examples):
        noun = random.choice(NOUNS)
        year = random.choice(YEARS)
        prompts.append(f"The {noun} lasted from the year {year} to " + year[:2])
    return prompts

def get_prompts_and_more(num_examples, seed):
    random.seed(seed)
    prompts = []
    for i in range(num_examples):
        noun = random.choice(NOUNS)
        year = random.choice(YEARS)
        idx = YEARS_BY_CENTRURY[year[:2]].index(year)
        ans = [y[2:] for y in YEARS_BY_CENTRURY[year[:2]][idx+1:]]
        obj = {"text": f"The {noun} lasted from the year {year} to " + year[:2],
               "01text": f"The {noun} lasted from the year {year[:2]+'01'} to " + year[:2],
               "target": ans}
        prompts.append(obj)

    return prompts

if __name__ == "__main__":
    prompts = get_prompts(2000, 0)
    with open("../dataset/prompts_greater_than.json", "w") as f:
        json.dump(prompts, f)
    print("data saved.")