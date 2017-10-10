measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
    {'city': '', 'temperature': 18.}
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

#a = vec.fit_transform(measurements).toarray()
a = vec.fit_transform(measurements)

print(a)
print(vec.get_feature_names())
