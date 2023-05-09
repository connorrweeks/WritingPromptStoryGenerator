import praw
import pandas as pd
from praw.models import MoreComments
from psaw import PushshiftAPI
import time
import random as r

# Define user agent
user_agent = "praw_scraper_1.0"

# Create an instance of reddit class
reddit = praw.Reddit(username="PredatorySquid",
                     password="1214connor",
                     client_id="Vvl8KZObiuQafMWItrlxvQ",
                     client_secret="6eiQOSLCkX865IXIVoInD6tXeAmbxA",
                     user_agent=user_agent
)
api = PushshiftAPI(reddit)

import datetime as dt
from datetime import datetime, date, timedelta

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def save_urls(y):
    urls = {}
    start_date = datetime(y, 1, 1)
    #end_date = datetime(2014, 1, 2)
    end_date = datetime(y+1, 1, 1)
    for single_date in daterange(start_date, end_date):

        start_epoch=int(single_date.timestamp())
        end_epoch=int((single_date + timedelta(days=1)).timestamp())

        day_posts = list(api.search_submissions(after=start_epoch,
                                before=end_epoch,
                                subreddit='writingprompts',
                                filter=['url','author', 'title', 'subreddit'],
                                limit=1000))
        print('\r', single_date.strftime("%Y-%m-%d"), len(day_posts), end='')

        for d in day_posts:
            urls[d.url] = d.num_comments
    #print(urls)
    print('\n', len(urls))
    f = open(f'./posts_urls_{y}.txt', 'w+')
    f.write('\n'.join([x + ' ' + str(urls[x]) for x in urls]))
    f.close()
    return urls

def find_all_posts():
    all_urls = {}
    for y in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:
        urls = save_urls(y)
        for u in urls:
            all_urls[u] = urls[u]
    print(len(all_urls))
    f = open(f'./posts_urls_all.txt', 'w+')
    f.write('\n'.join([x + ' ' + str(all_urls[x]) for x in all_urls]))
    f.close()

def print_all_posts():
    urls = {}
    with open(f'./posts_urls_all.txt') as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            url, num = l.split(' ')
            if(url in urls):
                print('Alreadly found')
                exit()
            urls[url] = int(num)
    rev = {}
    total = 0
    for url in urls:
        total += urls[url]
        if(urls[url] not in rev):
            rev[urls[url]] = 0
        rev[urls[url]] += 1
    print(len(rev))
    real_total = 0
    for i in reversed(range(1, 10000)):
        if(i in rev):
            real_total += (i-1) * rev[i]
            print(f'{i} - {rev[i]}')
    return [x for x in urls if urls[x] > 50]
    print('total', total)
    print('real_total', real_total)

def read_all_posts():
    urls = []
    with open(f'./posts_urls_all.txt') as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            url, num = l.split(' ')
            if(int(num) > 1 and url != 'self'):
                urls.append(url)
    return [x for x in urls]

def download_stories(subreddit, urls, group_num, group_size):
    group_urls = urls[group_num * group_size:(1+group_num) * group_size]

    submission_ids = [x.split('/')[-3] for x in group_urls]

    # creating lists for storing scraped data
    prompts, stories, scores, prompt_links, story_scores = [], [], [], [], []

    i, j, deleted, short = 0, 0, 0, 0
    #for submission in subreddit.top(limit=None):
    t_0 = time.perf_counter()
    for submission_id in submission_ids: # looping over posts and scraping it
        j += 1
        if(submission_id.strip() == ''): continue

        try:
            submission = reddit.submission(submission_id)
        except ValueError as e:
            print('\n', e, "\nsubmission_id", submission_id)
            exit()

        for top_level_comment in submission.comments[1:]: # looping over comments excluding read me
            if isinstance(top_level_comment, MoreComments):
                continue
            if top_level_comment.body == '[deleted]':
                deleted += 1
                continue
            if len(top_level_comment.body) < 300:
                short += 1
                continue
            i += 1

            t_1 = time.perf_counter()
            time_per = (t_1 - t_0) / j
            time_for_group = (group_size - j) * time_per
            time_for_all = (len(urls) - (group_num * group_size) + (group_size - j)) * time_per

            print(f'\rGroup<{group_num}/{int(len(urls) / group_size) + 1}>  prompts:{j}/{group_size}   stories:{i} deleted:{deleted} short:{short} group_time:{time_for_group:.2f}  total_time:{time_for_all:.2f}  {time_for_all/3600:.2f}hr------', end='')

            prompt_links.append(submission_id)
            prompts.append(submission.title)
            stories.append(top_level_comment.body)
            story_scores.append(top_level_comment.score)
            scores.append(submission.score)

    # creating dataframe for displaying scraped data
    df = pd.DataFrame()
    df['scores'] = scores
    df['prompts'] = prompts
    df['stories'] = stories
    df['prompt_links'] = prompt_links
    df['story_scores'] = story_scores

    df.to_csv(f'./data/group_{group_num}.csv')

    print()
    print(df.shape)
    print(df.head(10))

def clean(story):
    return story

def main():
    # Create sub-reddit instance
    #subreddit_name = "writingprompts"
    #subreddit = reddit.subreddit(subreddit_name)
    #
    df = pd.read_csv(f'./data/train.csv')
    print(df.head())

    stories = df['scores'].to_list()
    t = 0
    for x in stories:
        if('this post has been removed.' in x):
            t += 1
    print(len(df))

    print(t)


def combine_into_csv():
    total = 0
    all_prompts, all_stories = [], []
    for g in range(372):
        df = pd.read_csv(f'./data/group_{g}.csv')
        prompts = df['prompts']
        stories = df['stories']
        for i, x in enumerate(prompts):
            if(x == '[deleted]'):
                continue
            if(type(stories[i]) == float):
                continue
            if('Welcome to the Prompt!' in stories[i]):
                continue
            all_prompts.append(x)
            all_stories.append(stories[i])
        total += len(df)
        print(f"\r{g}/372 - {len(all_prompts)}", end='')

    n = len(all_prompts)
    ind = list(range(n))
    r.shuffle(ind)

    all_stories = [all_stories[ind[i]] for i in range(n)]
    all_prompts = [all_prompts[ind[i]] for i in range(n)]

    cut_off = int(n * 0.90)
    cut_off2 = int(n * 0.95)
    df = pd.DataFrame()
    df['scores'] = all_stories[:cut_off]
    df['prompts'] = all_prompts[:cut_off]
    df.to_csv('./data/train.csv')

    df = pd.DataFrame()
    df['scores'] = all_stories[cut_off:cut_off2]
    df['prompts'] = all_prompts[cut_off:cut_off2]
    df.to_csv('./data/val.csv')

    df = pd.DataFrame()
    df['scores'] = all_stories[cut_off2:]
    df['prompts'] = all_prompts[cut_off2:]
    df.to_csv('./data/test.csv')

    print(df.head())

    #group_size = 1000
    #urls = read_all_posts()
    #print("Number of urls:", len(urls))
    #print("Total Groups:", int(len(urls) / group_size) + 1)
    #for group_num in range(302, int(len(urls) / group_size) + 1):
    #    download_stories(subreddit, urls, group_num, group_size)

if(__name__ == '__main__'):
    main()
