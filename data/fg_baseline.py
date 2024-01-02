import pandas as pd
from tqdm import tqdm
def summary_time(df):
    result = df.groupby('id')['action_time'].sum().reset_index()
    result.rename(columns={'action_time': 'summary_time'}, inplace=True)
    return result
def start_pause(df):
    result = df.groupby('id')['down_time'].min().reset_index()
    result.rename(columns={'down_time': 'start_pause'}, inplace=True)
    return result
def enter_click(df):
    copy_df = df
    copy_df['enter_click'] = (copy_df['down_event'] == 'Enter')
    copy_df = copy_df.groupby('id')['enter_click'].sum().reset_index()
    return copy_df
def space_click(df):
    copy_df = df
    copy_df['space_click'] = (copy_df['down_event'] == 'Space')
    copy_df = copy_df.groupby('id')['space_click'].sum().reset_index()
    return copy_df
def backspace_click(df):
    copy_df = df
    copy_df['backspace_click'] = (copy_df['down_event'] == 'Backspace')
    copy_df = copy_df.groupby('id')['backspace_click'].sum().reset_index()
    return copy_df
def symbol_length(df):
    result = df.groupby('id')['cursor_position'].max().reset_index()
    result.rename(columns={'cursor_position': 'symbol_length'}, inplace=True)
    return result
def text_length(df):
    result = df.groupby('id')['word_count'].max().reset_index()
    return result
def nonproduction_feature(df):
    result = df.groupby('id')['activity'].apply(lambda x: (x == 'Nonproduction').mean() * 100).reset_index()
    result.rename(columns={'activity': 'nonproduction_feature'}, inplace=True)
    return result
def input_feature(df):
    result = df.groupby('id')['activity'].apply(lambda x: (x == 'Input').mean() * 100).reset_index()
    result.rename(columns={'activity': 'input_feature'}, inplace=True)
    return result
def remove_feature(df):
    result = df.groupby('id')['activity'].apply(lambda x: (x == 'Remove/Cut').mean() * 100).reset_index()
    result.rename(columns={'activity': 'remove_feature'}, inplace=True)
    return result
def mean_action_time(df):
    result = df.groupby('id')['action_time'].mean().reset_index()
    result.rename(columns={'action_time': 'mean_action_time'}, inplace=True)
    return result
def replace_feature(df):
    result = df[df['activity'] == 'Replace'].groupby('id').size().reset_index(name='replace_feature')
    return result
def text_change_unique(df):
    result = df.groupby('id')['text_change'].nunique().reset_index()
    result.rename(columns={'text_change': 'tch_unique'}, inplace=True)
    return result
def sentence_size_feature(df):
    result = df[(df['text_change'] == '.') & (df['down_event'] != 'Backspace')].groupby('id').size().reset_index(name = 'number_sentence')
    return result
def get_mm_time(df):
    mm = {'id': [], 'mm_time': []}
    groups = df.groupby('id')
    for gr in tqdm(groups):
        g = gr[1]
        k = gr[0]
        de = list(g.down_event)
        dt = list(g.down_time)
        ut = list(g.up_time)
        start_time = -1
        t = -1
        for i in range(len(de)):
            if start_time == -1 and ('Audio' in de[i] or 'Media' in de[i]):
                start_time = dt[i]
                t = ut[i] - dt[i]
            elif 'Audio' in de[i] or 'Media' in de[i]:
                t = ut[i] - start_time
        if start_time == -1:
            mm['mm_time'].append(0)
        else:
            mm['mm_time'].append(t)

        mm['id'].append(k)
    return pd.DataFrame(mm)

def getDataset(train_df):
    new_df = summary_time(train_df)

    functions = [
        start_pause, enter_click, space_click,
        backspace_click, symbol_length, text_length, nonproduction_feature,
        input_feature, remove_feature, mean_action_time,replace_feature,text_change_unique, sentence_size_feature,
        get_mm_time
    ]

    for func in functions:
        result_df = func(train_df)
        new_df = pd.merge(new_df, result_df, on='id', how='outer')

    return new_df

train_df = pd.read_csv('./dataset/original/train_logs.csv')
train_scores = pd.read_csv('./dataset/original/train_scores.csv')
# train_df = pd.read_csv("./dataset/train.csv")
train_df = getDataset(train_df)

train_df = pd.merge(train_df, train_scores, on='id', how='outer')
train_df['replace_feature'].fillna(0,inplace = True)
train_df['number_sentence'].fillna(0,inplace = True)

train_df.to_csv('./dataset/features.csv', index=False)