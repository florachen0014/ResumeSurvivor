import os
import re

# resume
import resume

RESUME_SECTIONS_GRAD = [
                    #experience
                    'work experience',
                    'experience',
                    'professional experience',
                    'accomplishments',
                    'education',
                    'interests',
                    'projects',
                    'publications',
                    'skills',
                    'certifications',
                    'objective',
                    'career objective',
                    'resume objective',
                    'summary',
                    'leadership'                  
                ]

def extract_text(pdf_path):
    res = resume.extract_text(pdf_path)
    res = res.encode('ascii', 'ignore').decode()
    return res

def get_res_sections(res):
    text_split = [re.sub('[^A-Za-z]',' ',i.lower()).strip() for i in res.split('\n')]
    text_split = [i for i in text_split if i != '']
    sections = []
    for line in text_split:
        if line in RESUME_SECTIONS_GRAD:
            sections.append((line, text_split.index(line)))
    res_sec = {}
    for i in range(0,len(sections)):
        sec = sections[i]
        if i == len(sections)-1:
            res_sec[sec[0]] = text_split[sec[1]+1:]
        else:
            res_sec[sec[0]] = text_split[sec[1]+1:sections[i+1][1]]
    return res_sec

def get_res_sections_from_path(pdf_path):
    res = extract_text(pdf_path)
    res_sec = get_res_sections(res)
    return res_sec