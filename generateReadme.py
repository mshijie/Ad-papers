#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import urllib

""" generate readme.md """

out_file = open('./README.md', 'w')

github_root = "https://github.com/mshijie/Ad-papers/blob/master/"
all_dir = sorted(os.listdir("./"))
for one_dir in all_dir:
    if os.path.isdir(one_dir) and not one_dir.startswith('.'):
        out_file.write("\n### " + one_dir+"\n")
        all_sub_files = os.listdir(one_dir)
        for one_file in all_sub_files:
            if not os.path.isdir(one_file) and not one_file.startswith('.'):
                out_file.write("* [" + ('.').join(one_file.split('.')[:-1]) + "]("+github_root + urllib.quote(one_dir.strip())+"/"
                               + urllib.quote(one_file.strip())+") <br />\n")

out_file.close()
