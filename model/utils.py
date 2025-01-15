import json, jsonlines
import os
from tqdm import tqdm
import re
import calendar
import inflect

class Alias():
    def __init__(self):
        
        self.ent2id = {}
        self.rel2id = {}
        self.id2aliases = {}
        
        with open("/data/ljx/roht/wikidata5m/wikidata5m_entity.txt", "r") as f:
            for line in tqdm(f.readlines()):
                line = line.split("\t")
                entid, aliases = line[0], line[1:]
                self.id2aliases[entid] = aliases
                for ent in aliases:
                    if ent not in self.ent2id:
                        self.ent2id[ent] = entid
                    elif len(aliases) > self.id2aliases[self.ent2id[ent]]:
                        self.ent2id[ent] = entid

        with open("/data/ljx/roht/wikidata5m/wikidata5m_relation.txt", "r") as f:
            for line in tqdm(f.readlines()):
                line = line.split("\t")
                relid, aliases = line[0], line[1:]
                self.id2aliases[relid] = aliases
                for rel in aliases:
                    self.rel2id[rel] = relid 
        
        self.P = inflect.engine()
    
    def get_aliases(self, s):
        if self.is_date(s):
            year, month, day = [int(x) for x in s.split("-")]
            return [s, "%s %d, %d" % (calendar.month_name[month], day, year), "%s %d, %d" % (calendar.month_abbr[month], day, year)]
        elif s in self.ent2id:
            return [s] + self.id2aliases[self.ent2id[s]][:5] 
        elif s in self.rel2id:
            return [s] + self.id2aliases[self.rel2id[s]][:5]
        elif len(re.findall(r"\d+", s))== 1:
            num = re.findall(r"\d+", s)[0]
            return [" "+s.replace(num, x) for x in self.num_aliases(num)]
        return [s]
    
    def num_aliases(self, s):
        res = [s, self.P.number_to_words(s)]
        if len(s) > 3:
            p = len(s) % 3
            parts = [s[max(0, i-3):i] for i in range(p, len(s)+1, 3)]
            res.extend([",".join(parts), ", ".join(parts)])
        return res

    def ent_aliases(self, s):
        return [s] + self.id2aliases[self.ent2id[s]][:5] if s in self.ent2id else [s]
    
    def is_date(self, s):
        if s.count("-") != 2:
            return False
        nums = s.split("-")
        if len(nums) != 3 or any((x.isdigit() == False) for x in nums):
            return False
        year, month, day = [int(x) for x in nums]
        if year <= 9999 and month >= 1 and month <= 12 and day >= 1 and day <= 31:
            return True
        else:
            return False
        
def answer_offset_in_context(answer_text, context_text):
    answer_text = answer_text.strip()
    context_text = context_text.strip()
    if f" {answer_text} " in context_text:
        return context_text.index(f" {answer_text} ")+1
    elif f" {answer_text}" in context_text:
        return context_text.index(f" {answer_text}")+1
    elif f"{answer_text} " in context_text:
        return context_text.index(f"{answer_text} ")
    elif answer_text in context_text:
        return context_text.index(answer_text)

    # If can't find case-specific occurrence, find first lower cased.
    answer_text = answer_text.lower()
    context_text = context_text.lower()
    if f" {answer_text} " in context_text:
        return context_text.index(f" {answer_text} ")+1
    elif f" {answer_text}" in context_text:
        return context_text.index(f" {answer_text}")+1
    elif f"{answer_text} " in context_text:
        return context_text.index(f"{answer_text} ")
    elif answer_text in context_text:
        return context_text.index(answer_text)

    return None

if __name__ == "__main__":
    a = Alias()
    print(a.get_alias("5123456 km"))