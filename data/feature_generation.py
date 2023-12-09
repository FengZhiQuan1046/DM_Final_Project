import pandas as pd
# from torch.utils.data import Dataset
import json
from tqdm import tqdm
from copy import copy
import re


if __name__ == '__main__':
    data = pd.read_csv("./dataset/train.csv")
    groups = [each[1] for each in data.groupby("id")]
    groups_dicts = []
    for i in tqdm(range(len(groups))):
        time_point = list(groups[i].down_time)
        down_event = list(groups[i].down_event)
        cursor = list(groups[i].cursor_position)
        change = list(groups[i].text_change)
        label = list(groups[i].score)[0]
        id = list(groups[i].id)[0]
        text = ""
        buf_text = ""
        state = ""
        select = ""
        select_range = [-1, -1]
        current_cursor = 0
        audio_start = 0
        media_start = 0
        audio_time = 0
        media_time = 0
        remove_amount = 0
        copy_amount = 0
        
        for j in range(len(down_event)):
            # if edit_list[j] in ['MediaTrackPrevious', 'Cancel', 'MediaTrackNext', 'PageUp', 'V', 'T', 'A', 'S', 
            #                     'AudioVolumeMute', 'PageDown', 'ScrollLock', 'Process', 'Unidentified', 'Middleclick', 'MediaPlayPause', 
            #                     'Dead','Insert', 'Alt', 'Rightclick', 'NumLock', 'End', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'Home','ContextMenu', 'Tab', 'Meta', 'Shift', 
            #                     'Control', 'Leftclick', 'ArrowRight', 'ArrowLeft', 'ArrowDown', 'AudioVolumeUp', 'AudioVolumeDown', 'ArrowDown', 
            #                     'ArrowUp', 'CapsLock']:
                # current_cursor = cursor[j]
            if state == "" and (down_event[j] in ['Space', 'Enter', 'Tab'] or down_event[j] in '.,;"-\'!?[]=abcdefghijklmnopqrstuvwxyz\\/<>0123456789$#@^&%*()+_-:`~{}|¿'):
                if down_event[j] == 'Space': tok = ' '
                elif down_event[j] == 'Enter': tok = '\n'
                else: tok = down_event[j]
                if '=>' in change[j]:
                    text = f"{text[:select_range[0]]}{text[select_range[1]:]}"
                    remove_amount += select_range[1] - select_range[0]
                    # print(change[j])
                    current_cursor = select_range[0]
                    select_range = [-1, -1]
                    select = ""
                text = f"{text[:current_cursor]}{tok}{text[current_cursor:]}"
                # current_cursor = cursor[j]
            elif down_event[j] in ['Backspace']:
                remove_amount += 1
                previous_cursor = current_cursor
                # current_cursor = cursor[j]
                text = f"{text[:current_cursor]}{text[previous_cursor:]}"
                # current_cursor = cursor[j]
            elif down_event[j] in ['Delete']:
                text = f"{text[:current_cursor]}{text[current_cursor+1:]}"
                # current_cursor = cursor[j]
            elif down_event[j] == 'Control':
                state = down_event[j]
            elif state == "Control" and down_event[j] == 'x':
                buf_text = copy(select)
                # buf_text = text[cursor[j] : cursor[j-1]]
                text = f"{text[:cursor[j]]}{text[cursor[j-1]:]}"
                state = ""
                
            elif state == "Control" and down_event[j] == 'a':
                select = copy(text)
                state = ""
                select_range = [0, len(text)]
            elif state == "Control" and down_event[j] == 'c':
                # print(down_event[j-4: j+1], cursor[j-4: j+1])
                buf_text = copy(select)
                state = ""
                copy_amount += len(buf_text.split())
            elif state == "Control" and down_event[j] == 'v':
                # print(edit_list[j], edit_list[j+1], cursor[j], cursor[j+1])
                if '=>' in change[j]:
                    text = f"{text[:select_range[0]]}{text[select_range[1]:]}"
                    remove_amount += select_range[1] - select_range[0]
                    # print(change[j])
                    select_range = [-1, -1]
                    select = ""

                select_range = [cursor[j], cursor[j-1]]
                select = copy(buf_text)
                text = f"{text[:current_cursor]}{buf_text}{text[current_cursor:]}"
                state = ""
            elif down_event[j] == 'Leftclick':
                # select = ""
                # if cursor[j-1] > cursor[j]:
                #     print(print(down_event[j-4: j+1], cursor[j-4: j+1]))
                select_range = [min(cursor[j-1],cursor[j]), max(cursor[j-1],cursor[j])]
                select = text[select_range[0]: select_range[1]]
            else: 
                if 'Audio' in down_event[j]:# in ['AudioVolumeMute', 'AudioVolumeUp', 'AudioVolumeDown']:
                    if audio_start == 0: audio_start = time_point[j]
                    audio_time = time_point[j] - audio_start
                if 'Media' in down_event[j]:# in ['MediaTrackPrevious', 'MediaTrackNext', 'MediaPlayPause']:
                    if media_start == 0: media_start = time_point[j]
                    media_time = time_point[j] - media_start
                state = ""
                # current_cursor = cursor[j]
                # print(edit_list[j], cursor[j-1], cursor[j], len(text))
            current_cursor = cursor[j]
        words_count = len([each for each in text.split() if len(each)>=1])
        edit_time = time_point[-1] - time_point[0]
        sentences = re.split(r'[.!?]', text)
        sentences = [each for each in sentences if len(each) >= 3]
        average_sentence_len = sum([len(each) for each in sentences])/len(sentences)
        paragraphs = re.split(r'[\n]', text)
        paragraphs = [each for each in paragraphs if len(each) >= 5]
        paragraph_num = len(paragraphs)
        max_mark_patterns_len = max([len(each) for each in text.split('q')])
        stop_rate = text.count('.')/(text.count('.') + text.count('!') + text.count('?')+1)
        groups_dicts.append({'id': id, 'words_count': words_count, 'remove_amount': remove_amount, 'copy_amount': copy_amount, 'average_sentence_len': average_sentence_len, 
                                'paragraph_num':paragraph_num, 'edit_time': edit_time, 'audio_time': audio_time, 'media_time': media_time, 
                                'stop_rate': stop_rate, 'max_mark_patterns_len': max_mark_patterns_len, 'text': text, 'label':label})
        with open('./dataset/features.json', 'w') as f:
            json.dump(groups_dicts, f)