#!/usr/bin/env python
# -*- coding: utf-8 -*-


wf = open('sample_images/categories.txt', 'wt')
with open('synset_words.txt', 'rt') as rf:
    line = rf.readline()
    while line:
        line = line.strip().split(' ', 1)[1]
        line = line.split(',', 1)[0].strip().replace(' ', '_')
        wf.write(line + "\n")
        line = rf.readline()
wf.close()
