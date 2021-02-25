# This script formats the raw bAbI stories.
# The stories remain the same but words become separated by space. Stories are
# not split nor merged. Periods are words. Answers with commas are one word,
# i.e. "n,w" is one word. Samples (i.e. stories and their questions) are
# separated by "\n". Supporting labels are removed.

import os
import pickle
import random

DATASETS = ["en-valid", "en-valid-10k"]
PARTITIONS = ["train", "valid", "test"]
DATA_PATH = ".tmp/tasks_1-20_v1-2/{}/qa{}_{}.txt"
OUTPUT_PATH = "data/catbAbI_v1.2/"

PAD = "<pad>"
EOS = "<eos>"


def parse_file_to_samples(path, task):
    """ returns [story:[str], (task_id:int, answer:str)] """
    f = open(path, "r")
    samples = []
    story = []
    for line in f:
        tid, text = line.lower().rstrip('\n').split(' ', 1)
        if tid == "1" and len(story) > 0:
            # new story begins, add the current story as sample if there is one
            samples.append((list(story), task))
            story = []
        if text.endswith('.'):
            # non-question, append words and period to the story so far
            words = text[:-1].split(' ')
            story.extend(words)
            story.append(".")
        else:
            # question line, append question and answer but remove support
            query, answer, _ = (x.strip() for x in text.split('\t'))
            query_words = query[:-1].split(' ')
            # copy the current story as it may continue for the next sample
            story.extend(query_words)
            story.append("?")
            story.append(answer)
    f.close()
    return samples


def read_task_files(dataset, partition):
    """ returns [story:[str], (task_id:int, answer:str)] """
    all_samples = []
    for task in list(range(1, 21)):
        s = parse_file_to_samples(
              path=DATA_PATH.format(dataset, task, partition),
              task=task)
        # print("task {}: {} samples".format(task, len(s)))
        all_samples.extend(s)
    return all_samples


def load_all_data():
    """ returns {str:{str:[[str], str]}}"""
    data = {}
    for dataset in DATASETS:
        data[dataset] = {}
        for partition in PARTITIONS:
            data[dataset][partition] = read_task_files(dataset, partition)
    return data


def get_vocabulary(data):
    """ returns [] """
    unique_words = [PAD, EOS]
    for d in DATASETS:
        for p in PARTITIONS:
            for sample in data[d][p]:
                for word in sample[0]:
                    if word in unique_words:
                        continue
                    else:
                        unique_words.append(word)
                if not sample[1] in unique_words:
                    unique_words.append(sample[1])

    word2idx = {w: i for i, w in enumerate(unique_words)}
    idx2word = {i: w for i, w in enumerate(unique_words)}
    return word2idx, idx2word


def write_files(data, path):
    for d in DATASETS:
        for p in PARTITIONS:
            random.shuffle(data[d][p])
            f = open(os.path.join(OUTPUT_PATH, "{}_{}.txt".format(d, p)), "w+")
            for sample in data[d][p]:
              story = " ".join(sample[0])
              task = sample[1]
              line = "{}\t{}\n".format(task, story)
              f.write(line)
            f.close()


def write_vocab(word2idx, idx2word, path):
    f = open(os.path.join(OUTPUT_PATH, "vocab.pkl"), "wb")
    pickle.dump([word2idx, idx2word], f)
    f.close()


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

    # process joint data
    all_data = load_all_data()
    write_files(all_data, OUTPUT_PATH)
    # vocab
    word2idx, idx2word = get_vocabulary(all_data)
    write_vocab(word2idx, idx2word, OUTPUT_PATH)
    print(f"catbAbI is ready at {OUTPUT_PATH}")
else:
    print(f"ouput directory {OUTPUT_PATH} already exists. Nothing was done.")
