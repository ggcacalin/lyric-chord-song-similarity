import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import textdistance as tdist
import ast
import nltk
import multiprocessing
import math
from joblib import Parallel, delayed
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


#Cleaning up initial data
data = pd.read_csv('all_chords_ids.csv')
#Drop irrelevant columns for analysis
data.drop(['Artist', 'Song', 'Language'], axis=1, inplace=True)
#Move unique identifier to be first column
data = data[['Song_id', 'Artist_id', 'Genre', 'Lyric', 'Chords', 'Final']]
#Clean bad rows and remove random rock songs until same number as pop
data.drop([174,884], inplace=True)
np.random.seed(42)
while (len(data['Genre'][456:]) > len(data['Genre'][:456])):
    row = np.random.randint(456, len(data['Genre']))
    if row in data.index:
        data.drop(row, inplace=True)
#Reindex song id
data = data.set_index('Song_id')
data = data.reset_index(drop=True)
#Remove artists with less than 3 songs
artist_ids = data['Artist_id'].to_numpy(dtype=int)
killrow_list = []
counter = 1
for i in range(1, len(artist_ids), 1):
    if artist_ids[i] == artist_ids[i-1]:
        counter+=1
    else:
        if counter < 3:
            for j in range(counter):
                killrow_list.append(i - 1 - j)
        counter = 1
data.drop(killrow_list, inplace=True)
data = data.reset_index(drop=True)
#Delete 0 duration pitches, clone it
basic_pitches = data['Chords'].tolist()
duration_list = []
for i in range(len(basic_pitches)):
    basic_pitches[i] = ast.literal_eval(basic_pitches[i])
for pitches in basic_pitches:
    kill_indices = []
    for i in range(len(pitches)):
        if pitches[i][1] == 0:
            kill_indices.append(pitches[i])
        else:
            #Remember unique non-zero durations
            if pitches[i][1] not in duration_list:
                duration_list.append(pitches[i][1])
    for i in kill_indices:
        pitches.remove(i)
#Remap durations to scaled integers
duration_dict = {}
min_duration = min(duration_list)
for duration in duration_list:
    duration_dict[duration] = int(duration / min_duration)
for pitches in basic_pitches:
    for i in range(len(pitches)):
        pitches[i] = (pitches[i][0], duration_dict[pitches[i][1]])
data.insert(3, 'Duration Chords', basic_pitches, True)
#Replace chords with basic pitches
basic_pitches = data['Duration Chords'].tolist()
new_chords = []
for pitches in basic_pitches:
    song_chords = []
    for i in range(len(pitches)):
        song_chords.append(pitches[i][0])
    new_chords.append(song_chords)
data.drop('Chords', axis=1, inplace=True)
data.insert(3, 'Chords', new_chords, True)
#Re-compute final
basic_pitches = data['Duration Chords'].tolist()
final_pitches = data['Final'].tolist()
for i in range(len(final_pitches)):
    final_pitches[i] = []
    for pair in basic_pitches[i]:
        for j in range(pair[1]):
            final_pitches[i].append(pair[0])
data.drop('Final', axis=1, inplace=True)
data.insert(4, 'Final', final_pitches, True)
data.to_csv("clean_full_lyrics.csv", index=False)


#Tokenizaiton
data = pd.read_csv('clean_full_lyrics.csv')
lyrics = data['Lyric'].to_numpy().tolist()
#Method to get part of speech
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

for i in range(len(lyrics)):
    print(i)
    lyrics[i] = word_tokenize(lyrics[i])
    lyrics[i] = [WordNetLemmatizer().lemmatize(token, get_wordnet_pos(token)) for token in lyrics[i]]
    lyrics[i] = ' '.join(str(t) for t in lyrics[i])
data.drop('Lyric', axis=1, inplace=True)
data.insert(4, 'Lyric', lyrics, True)
data.to_csv('tokenized_lyrics.csv', index=False)


#TF-IDF, pruning, lyrical similarity
data = pd.read_csv('tokenized_lyrics.csv')
x=data['Lyric']
vectorizer = TfidfVectorizer(min_df=0.05)
X = vectorizer.fit_transform(x.values.astype(str))
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
data = pd.concat([data, tfidf_df], axis=1)
#Compute all pairwise similarities
n = len(x.to_numpy())
SL = np.ones((n, n))
transformed = X.toarray()
for i in range(n):
    for j in range(i+1, n):
        SL[i, j] = np.dot(transformed[i], transformed[j]) / \
            (np.linalg.norm(transformed[i]) * np.linalg.norm(transformed[j]))
        SL[j, i] = SL[i, j]
lyrical_similarities = pd.DataFrame(SL)
lyrical_similarities.to_csv('l_similarities.csv', index=False)
data.to_csv('processed_data.csv')


#Musical similarity, with and without duration enhancement
data = pd.read_csv('processed_data.csv')

x = data['Chords'].tolist()
for i in range(len(x)):
    x[i] = ast.literal_eval(x[i])
n = len(data['Chords'])

# change to use multiple cores if possible
num_cores = multiprocessing.cpu_count()
distances = Parallel(n_jobs=num_cores - 1)(delayed(tdist.levenshtein.normalized_similarity)(
    x[i], x[j]) for i in tqdm(range(n)) for j in range(i, n))
SM = np.ones((n, n))
dist_idx = 0
for i in range(n):
    for j in range(i, n):
        SM[i][j] = distances[dist_idx]
        SM[j][i] = distances[dist_idx]
        dist_idx += 1
SM = pd.DataFrame(SM)
SM.to_csv('m_similarities.csv', index=False)

x = data['Final'].tolist()
for i in range(len(x)):
    x[i] = ast.literal_eval(x[i])
n = len(data['Chords'])

# change to use multiple cores if possible
num_cores = multiprocessing.cpu_count()
distances = Parallel(n_jobs=num_cores - 1)(delayed(tdist.levenshtein.normalized_similarity)(
    x[i], x[j]) for i in tqdm(range(n)) for j in range(i, n))
SM = np.ones((n, n))
dist_idx = 0
for i in range(n):
    for j in range(i, n):
        SM[i][j] = distances[dist_idx]
        SM[j][i] = distances[dist_idx]
        dist_idx += 1
SM = pd.DataFrame(SM)
SM.to_csv('SM_levenshtein.csv', index=False)

lyrical_similarities = pd.read_csv('l_similarities.csv')
chord_SM = pd.read_csv('m_similarities.csv')
final_SM = pd.read_csv('SM_levenshtein.csv')

def research_1(SL, SM):
    #Compute averages
    n = len(SL)
    relevant_SL = []
    relevant_SM = []
    for i in range(n-1):
        for j in range(i+1, n):
            relevant_SL.append(SL[i][j])
            relevant_SM.append(SM[i][j])
    relevant_SL = np.array(relevant_SL)
    relevant_SM = np.array(relevant_SM)
    mean_SL = np.mean(relevant_SL)
    mean_SM = np.mean(relevant_SM)
    #Compute correlations
    corr = np.dot(relevant_SL - mean_SL, relevant_SM - mean_SM) /\
           math.sqrt(np.sum(np.square(relevant_SL-mean_SL))*np.sum(np.square(relevant_SM-mean_SM)))
    #Test significance
    t_score = corr * math.sqrt(len(relevant_SL) - 2) / math.sqrt(1 - corr * corr)
    p_value = stats.t.sf(abs(t_score), df = len(relevant_SL) - 2) * 2
    return corr, p_value

def research_2(SL, SM, genres):
    #Compute averages
    n = len(SL)
    pop_SL = []
    pop_SM = []
    rock_SL = []
    rock_SM = []
    for i in range(n-1):
        for j in range(i+1, n):
            if genres[i] == 'Pop' and genres[j] == 'Pop':
                pop_SL.append(SL[i][j])
                pop_SM.append(SM[i][j])
            if genres[i] == 'Rock' and genres[j] == 'Rock':
                rock_SL.append(SL[i][j])
                rock_SM.append(SM[i][j])
    pop_SL = np.array(pop_SL)
    pop_SM = np.array(pop_SM)
    mean_pop_SL = np.mean(pop_SL)
    mean_pop_SM = np.mean(pop_SM)
    rock_SL = np.array(rock_SL)
    rock_SM = np.array(rock_SM)
    mean_rock_SL = np.mean(rock_SL)
    mean_rock_SM = np.mean(rock_SM)
    #Compute correlations
    pop_corr = np.dot(pop_SL - mean_pop_SL, pop_SM - mean_pop_SM) /\
           math.sqrt(np.sum(np.square(pop_SL-mean_pop_SL))*np.sum(np.square(pop_SM-mean_pop_SM)))
    t_score = pop_corr * math.sqrt(len(pop_SL) - 2) / math.sqrt(1 - pop_corr * pop_corr)
    p_value_pop = stats.t.sf(abs(t_score), df=len(pop_SL) - 2) * 2
    rock_corr = np.dot(rock_SL - mean_rock_SL, rock_SM - mean_rock_SM) /\
           math.sqrt(np.sum(np.square(rock_SL-mean_rock_SL))*np.sum(np.square(rock_SM-mean_rock_SM)))
    t_score = rock_corr * math.sqrt(len(rock_SL) - 2) / math.sqrt(1 - rock_corr * rock_corr)
    p_value_rock = stats.t.sf(abs(t_score), df=len(rock_SL) - 2) * 2
    #Test significance
    z_pop = 0.5 * math.log((1 + pop_corr)/(1 - pop_corr))
    z_rock = 0.5 * math.log((1 + rock_corr)/(1 - rock_corr))
    z_score = (z_pop - z_rock) / math.sqrt(1/(len(pop_SL) - 3) + 1/(len(rock_SL) - 3))
    p_value = stats.norm.sf(abs(z_score)) * 2
    return (pop_corr, p_value_pop), (rock_corr, p_value_rock), p_value

def research_3(SL, SM, genres, artist_ids, alpha):
    n = len(SL)
    pop_same_SL = {}; rock_same_SL = {}
    pop_diff_SL = {}; rock_diff_SL = {}
    pop_same_SM = {}; rock_same_SM = {}
    pop_diff_SM = {}; rock_diff_SM = {}
    #Fill-in-similarity-lists-challenge (impossible)
    for i in range(n-1):
        for j in range(i+1, n):
            if genres[i] == 'Pop' and genres[j] == 'Pop':
                if artist_ids[i] == artist_ids[j]:
                    a_id = artist_ids[i]
                    #Create list of artist's similarities if it isn't there
                    if a_id not in pop_same_SL:
                        pop_same_SL[a_id] = []
                        pop_same_SM[a_id] = []
                    pop_same_SL[a_id].append(SL[i][j])
                    pop_same_SM[a_id].append(SM[i][j])
                else:
                    #Take care of both combinations, as artist j won't be coupled with i again
                    # (but rather only with further artists)
                    ai_id = artist_ids[i]
                    aj_id = artist_ids[j]
                    # Create list of artist's similarities if it isn't there
                    if ai_id not in pop_diff_SL:
                        pop_diff_SL[ai_id] = []
                        pop_diff_SM[ai_id] = []
                    if aj_id not in pop_diff_SL:
                        pop_diff_SL[aj_id] = []
                        pop_diff_SM[aj_id] = []
                    pop_diff_SL[ai_id].append(SL[i][j])
                    pop_diff_SM[ai_id].append(SM[i][j])
                    #Symmetry makes order of indices irrelevant for SL/SM
                    pop_diff_SL[aj_id].append(SL[i][j])
                    pop_diff_SM[aj_id].append(SM[i][j])
            if genres[i] == 'Rock' and genres[j] == 'Rock':
                if artist_ids[i] == artist_ids[j]:
                    a_id = artist_ids[i]
                    #Create list of artist's similarities if it isn't there
                    if a_id not in rock_same_SL:
                        rock_same_SL[a_id] = []
                        rock_same_SM[a_id] = []
                    rock_same_SL[a_id].append(SL[i][j])
                    rock_same_SM[a_id].append(SM[i][j])
                else:
                    # Take care of both combinations, as artist j won't be coupled with i again
                    # (but rather only with further artists)
                    ai_id = artist_ids[i]
                    aj_id = artist_ids[j]
                    # Create list of artist's similarities if it isn't there
                    if ai_id not in rock_diff_SL:
                        rock_diff_SL[ai_id] = []
                        rock_diff_SM[ai_id] = []
                    if aj_id not in rock_diff_SL:
                        rock_diff_SL[aj_id] = []
                        rock_diff_SM[aj_id] = []
                    rock_diff_SL[ai_id].append(SL[i][j])
                    rock_diff_SM[ai_id].append(SM[i][j])
                    #Symmetry makes order of indices irrelevant for SL/SM
                    rock_diff_SL[aj_id].append(SL[i][j])
                    rock_diff_SM[aj_id].append(SM[i][j])
    #Computing the correlations
    pop_same_corrs = []; pop_diff_corrs = []
    rock_same_corrs = []; rock_diff_corrs = []
    for a_id in np.unique(artist_ids):
        if a_id in pop_same_SL:
            corr = np.dot(pop_same_SL[a_id] - np.mean(pop_same_SL[a_id]),
                          pop_same_SM[a_id] - np.mean(pop_same_SM[a_id])) /\
           math.sqrt(np.sum(np.square(pop_same_SL[a_id] - np.mean(pop_same_SL[a_id]))*
                            np.sum(np.square(pop_same_SM[a_id] - np.mean(pop_same_SM[a_id])))))
            pop_same_corrs.append(corr)
        if a_id in pop_diff_SL:
            corr = np.dot(pop_diff_SL[a_id] - np.mean(pop_diff_SL[a_id]),
                          pop_diff_SM[a_id] - np.mean(pop_diff_SM[a_id])) / \
                   math.sqrt(np.sum(np.square(pop_diff_SL[a_id] - np.mean(pop_diff_SL[a_id])) *
                                    np.sum(np.square(pop_diff_SM[a_id] - np.mean(pop_diff_SM[a_id])))))
            pop_diff_corrs.append(corr)
        if a_id in rock_same_SL:
            corr = np.dot(rock_same_SL[a_id] - np.mean(rock_same_SL[a_id]),
                          rock_same_SM[a_id] - np.mean(rock_same_SM[a_id])) / \
                   math.sqrt(np.sum(np.square(rock_same_SL[a_id] - np.mean(rock_same_SL[a_id])) *
                                    np.sum(np.square(rock_same_SM[a_id] - np.mean(rock_same_SM[a_id])))))
            rock_same_corrs.append(corr)
        if a_id in rock_diff_SL:
            corr = np.dot(rock_diff_SL[a_id] - np.mean(rock_diff_SL[a_id]),
                          rock_diff_SM[a_id] - np.mean(rock_diff_SM[a_id])) / \
                   math.sqrt(np.sum(np.square(rock_diff_SL[a_id] - np.mean(rock_diff_SL[a_id])) *
                                    np.sum(np.square(rock_diff_SM[a_id] - np.mean(rock_diff_SM[a_id])))))
            rock_diff_corrs.append(corr)
    #Significance tests
    pop_results = []
    pop_diffs = []
    rock_results = []
    rock_diffs = []
    #Runs through pop artists
    for i in range(len(pop_same_corrs)):
        z_same = 0.5 * math.log((1 + pop_same_corrs[i])/(1 - pop_same_corrs[i]))
        z_diff = 0.5 * math.log((1 + pop_diff_corrs[i])/(1 - pop_diff_corrs[i]))
        z_score = (z_same - z_diff) / math.sqrt(1 / (len(pop_same_SL) - 3) + 1 / (len(pop_diff_SL) - 3))
        p_value = stats.norm.sf(abs(z_score)) * 2
        if p_value < alpha:
            pop_results.append(1)
            pop_diffs.append(pop_same_corrs[i] - pop_diff_corrs[i])
        else:
            pop_results.append(0)
    #Runs through rock artists
    for i in range(len(rock_same_corrs)):
        z_same = 0.5 * math.log((1 + rock_same_corrs[i])/(1 - rock_same_corrs[i]))
        z_diff = 0.5 * math.log((1 + rock_diff_corrs[i])/(1 - rock_diff_corrs[i]))
        z_score = (z_same - z_diff) / math.sqrt(1 / (len(rock_same_SL) - 3) + 1 / (len(rock_diff_SL) - 3))
        p_value = stats.norm.sf(abs(z_score)) * 2
        if p_value < alpha:
            rock_results.append(1)
            rock_diffs.append(rock_same_corrs[i] - rock_diff_corrs[i])
        else:
            rock_results.append(0)
    #Test significance of difference in proportions
    pop_phat = np.sum(pop_results) / len(pop_results)
    rock_phat = np.sum(rock_results) / len(rock_results)
    z_score = (pop_phat - rock_phat) / math.sqrt(pop_phat*(1 - rock_phat)*(1/len(pop_results) + 1/len(rock_results)))
    p_value = p_value = stats.norm.sf(abs(z_score)) * 2
    return (pop_phat, np.mean(pop_diffs)), (rock_phat, np.mean(rock_diffs)), p_value

#Remove songs whose artist has less than 4 songs
artist_ids = data['Artist_id'].to_numpy(dtype=int)
killrow_list = []
counter = 1
for i in range(1, len(artist_ids), 1):
    if artist_ids[i] == artist_ids[i-1]:
        counter+=1
    else:
        if counter < 4:
            for j in range(counter):
                killrow_list.append(i - 1 - j)
        counter = 1
data.drop(killrow_list, inplace=True)
lyrical_similarities.drop(killrow_list, inplace=True)
lyrical_similarities.drop(lyrical_similarities.iloc[:, killrow_list], axis=1, inplace=True)
chord_SM.drop(killrow_list, inplace=True)
chord_SM.drop(chord_SM.iloc[:, killrow_list], axis=1, inplace=True)
final_SM.drop(killrow_list, inplace =True)
final_SM.drop(final_SM.iloc[:, killrow_list], axis=1, inplace=True)
data = data.reset_index(drop=True)
lyrical_similarities = lyrical_similarities.reset_index(drop=True)
lyrical_similarities.columns = range(lyrical_similarities.columns.size)
chord_SM = chord_SM.reset_index(drop=True)
chord_SM.columns = range(chord_SM.columns.size)
final_SM = final_SM.reset_index(drop=True)
final_SM.columns = range(final_SM.columns.size)

chord_SM = chord_SM.to_numpy()
final_SM = final_SM.to_numpy()

print(research_1(lyrical_similarities, chord_SM))
print(research_1(lyrical_similarities, final_SM))
print('-----------------------------------------------------------------------')
print(research_2(lyrical_similarities, chord_SM, data['Genre'].tolist()))
print(research_2(lyrical_similarities, final_SM, data['Genre'].tolist()))
print('-----------------------------------------------------------------------')
print(research_3(lyrical_similarities, chord_SM, data['Genre'].tolist(), data['Artist_id'].tolist(), 0.05))
print(research_3(lyrical_similarities, final_SM, data['Genre'].tolist(), data['Artist_id'].tolist(), 0.05))

