#libraries
import pandas as pd
import numpy as np

import re

import requests
from bs4 import BeautifulSoup
import resume

def get_url(job_name, location):
    job_name = job_name.replace(' ','%20')
    location = location.replace(' ','%20')
    url = 'https://www.indeed.com/jobs?q=' + job_name + '&l=' + location
    return url

def extract_results(url):
    try:
        page = requests.get(url)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content,'html.parser')
            results = soup.find(id='resultsBody')
            return results
        else:
            print('An error occurred.')
    except:
        print('Cannot access the website.')

def get_page_count(url):
    try:
        results = extract_results(url)
        pagecount_text = results.find('div',id='searchCountPages').text
        index_of = pagecount_text.index('of')
        num_job = re.sub('[A-Za-z]','',pagecount_text[index_of+3:]).strip()
        num_job = int(num_job)
        num_page = int(num_job/10)
        return num_page, num_job
    except:
        return 0,0

def get_pages(url):
    try:
        num_page, num_job = get_page_count(url)
        pages = [url]
        for i in range(1,num_page+1):
            pages.append(url+'&start='+str(i*10))
        return pages
    except:
        return None

# def get_description(job):
#     qual_list = ['requirements','qualifications','required ','what you ll']
#     description = []
#     for p in job.split('\n'):
#         description += [re.sub('[^A-Za-z]',' ',s).strip().lower() for s in sent_tokenize(p)]
#     for desc in description:
#         if any([qual in desc for qual in qual_list]):
#             end_index = description.index(desc)
#             break
#         elif desc == 'skill':
#             end_index = description.index(desc)
#         else:
#             end_index = len(description)
#     job_description = ' '.join(description[:end_index])
#     return job_description

def get_skills(job):
	return '|'.join(resume.extract_skills(job))

def get_jobs(url, limit = 50):
    try:
        count = 0
        job_dict = {
            'title':[],
            'company':[],
            'location':[],
            'url':[],
            'description':[],
            # 'extracted_description':[],
            'skill':[]
        }
        pages = get_pages(url)
        num_page, num_job = get_page_count(url)
        limit = min(limit, num_job)
        for page in pages:
            results = extract_results(page)
            jobs = results.find_all('div',class_='jobsearch-SerpJobCard unifiedRow row result')
            for job in jobs:
                if count == limit:
                    break
                job_title = job.find('h2',class_='title')\
                            .find('a',{'data-tn-element':'jobTitle'}).text.replace('\n','')
                job_url = job.find('h2',class_='title')\
                          .find('a',{'data-tn-element':'jobTitle'})['href']
                job_url = 'https://www.indeed.com'+job_url
                company = job.find('span',class_='company').text.replace('\n','')
                location = job.find('div',class_='recJobLoc')['data-rc-loc'].replace('\n','')

                #Access job url page
                job_page = requests.get(job_url)
                if job_page.status_code == 200:
                    job_soup = BeautifulSoup(job_page.content,'html.parser')
                    job_description = job_soup.find('div',class_='jobsearch-jobDescriptionText')
                    job_description = re.sub(r'<.+?>','\n',str(job_description))
                else:
                    job_description = 'Cannot access the website.'
                job_dict['title'].append(job_title)
                job_dict['company'].append(company)
                job_dict['location'].append(location)
                job_dict['url'].append(job_url)
                job_dict['description'].append(job_description)
                # job_dict['extracted_description'].append(get_description(job_description))
                job_dict['skill'].append(get_skills(job_description))
                count += 1
        return job_dict
    except:
        return {'Error':'Cannot access the website.'}

def get_indeed_job(job_name, location, limit = 50):
    url = get_url(job_name, location)
    job_dict = get_jobs(url, limit = limit)
    return job_dict