#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import simplejson as json
from mojimoji import han_to_zen

jsonized = open("livedoor_news_corpus.json", "wt")
jsonized.write("[\n")
for i, path in enumerate(glob.glob("text/*/*.txt")):
    bname = os.path.splitext(os.path.basename(path))[0]
    story_dic = dict()
    story_dic['iid'] = str(i + 1)
    story_dic['sid'] = bname.split('-')[-1]
    if story_dic['sid'] == 'LICENSE':
        continue
    story_dic['category'] = '_'.join(bname.split('-')[:-1])
    with open(path, "rt") as rf:
        story = ""
        j = 0
        rl = rf.readline()
        while rl:
            if j in [0, 1, 3]:
                j += 1
                rl = rf.readline()
                continue
            rl = han_to_zen(rl.strip())
            if j == 2:
                story_dic['title'] = rl
            else:
                story += rl
            j += 1
            rl = rf.readline()
        story_dic['body'] = story
    if i > 0:
        jsonized.write(",\n")
    json.dump(story_dic, jsonized, ensure_ascii=False)
jsonized.write("\n]\n")
jsonized.close()
