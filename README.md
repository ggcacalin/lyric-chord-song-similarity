# lyric-chord-song-similarity

Project processing scraped data from [ultimate-guitar.com](https://www.ultimate-guitar.com/) to analyse the pairwise similarity between songs in different genres.

Similarity is computed over lyrics and chords. Cosine similarity over TF-IDF vectors is used for the lemmatized lyrics, and string edit distance is used over the ABC-notation chords.

Fast script uses the pre-generated CSVs in the repo, full script re-generates all data needed aside from the initial dataframe.
