import io
import os
import re

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFSyntaxError

# import spacy
# from spacy.matcher import Matcher

def extract_text(file_path): 
    text = ''
    for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    text = text.encode('ascii', 'ignore').decode()
    return text

def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files
    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    '''
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    if not isinstance(pdf_path, io.BytesIO):
        # extract text from local pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        # codec='utf-8',
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                return
    else:
        # extract text from remote pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    # codec='utf-8',
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return

# Education Degrees
EDUCATION = [
            'BE','B.E.', 'B.E', 'BS', 'B.S', 
            'ME', 'M.E', 'M.E.', 'MS', 'M.S', 
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII','Masters','Bachelors'
        ]

def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return str(education)

def extract_entity_sections(text):
    '''
    Helper function to extract resume sections from resumes.
    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    RESUME_SECTIONS = {
    'objective':[
                'objective',
                'career objective',
                'resume objective'
                ],
    'summary':['summary'],
    'experience':[
                'experience',
                'work experience',
                'professional experience'
                ],
    'accomplishments':['accomplishment','accomplishments'],
    'education':['education'],
    'projects':['projects','selected projects'],
    'publications':['publications'],
    'skills':['skills','technical skills','soft skills','certifications'],
    'interests':['interests'],
    'leadership':['leadership']
    }
    
    #see which sections are in a resume
    text_split = [re.sub('[^A-Za-z]',' ',i.lower()).strip() for i in text.split('\n')]
    text_split = [i for i in text_split if i != '']
    sections = []
    for line in text_split:
        for sec, names in RESUME_SECTIONS.items():
            if line in names:
                sections.append((sec, text_split.index(line)))

    #find resume sections
    res_sec = {}
    for i in range(0,len(sections)):
        sec = sections[i]
        if i == len(sections)-1:
            res_sec[sec[0]] = text_split[sec[1]+1:]
        else:
            res_sec[sec[0]] = text_split[sec[1]+1:sections[i+1][1]]
    return res_sec


def get_obj_exp(res):
    res_sec = extract_entity_sections(res)

    obj = ' '.join(res_sec['objective'])
    exp = ' '.join(res_sec['experience'])

    obj_exp = obj + ' ' + exp

    return obj_exp