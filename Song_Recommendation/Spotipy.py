import spotipy
import spotipy.oauth2 as oauth2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
client_id = 'ab411b28199d4ac0a55db7fc7af98521'
client_secret =  '13e175e5efc54993afb443a162aa54b5'
redirect_uri = 'http://localhost:8080/callback'
playlist_ids = {
    'Angry': '37i9dQZF1EIgNZCaOGb0Mi',
    'Disgusted': '37i9dQZF1E8KEaf5o7wGZB',
    'Fearful': '37i9dQZF1EIfMwRYymgnLH',
    'Happy': '37i9dQZF1EVJSvZp5AOML2',
    'Neutral': '4PFwZ4h1LMAOwdwXqvSYHd',
    'Sad': '37i9dQZF1DWSqBruwoIXkA',
    'Surprised': '7vatYrf39uVaZ8G2cVtEik'
}
def get_track_data(sp, playlist_id):
    track_data = []
    results = sp.playlist_tracks(playlist_id)
    for item in results['items']:
        track = item['track']
        name = track['name']
        album = track['album']['name']
        artist = track['album']['artists'][0]['name']
        emotion = playlist_ids_inv[playlist_id]
        track_data.append({'Name': name, 'Album': album, 'Artist': artist, 'Emotion': emotion})
    return pd.DataFrame(track_data)
def train_classifier(track_data):
    X = track_data[['Name', 'Album', 'Artist']]
    y = track_data['Emotion']
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    return encoder, classifier
if __name__ == "__main__":
    auth_manager = spotipy.oauth2.SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    playlist_ids_inv = {v: k for k, v in playlist_ids.items()}
    all_track_data = pd.DataFrame()
    for playlist_id in playlist_ids.values():
        track_data = get_track_data(sp, playlist_id)
        all_track_data = pd.concat([all_track_data, track_data], ignore_index=True)
    encoder, classifier = train_classifier(all_track_data)
    encoder_filename = 'encoder.pkl'
    classifier_filename = 'classifier.pkl'
    pd.to_pickle(encoder, encoder_filename)
    pd.to_pickle(classifier, classifier_filename)

