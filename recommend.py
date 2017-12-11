"""A Yelp-powered Restaurant Recommendation Program"""

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main, trace, interact
from utils import distance, mean, zip, enumerate, sample
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #



def find_closest(location, centroids):
    """Return the item in CENTROIDS that is closest to LOCATION. If two
    centroids are equally close, return the first one.
    >>> find_closest([3, 4], [[0, 0], [2, 3], [4, 3], [5, 5]])
    [2, 3]
    """

    "*** YOUR CODE HERE ***"
    return min(centroids, key = lambda x: distance(location, x))
    # the lambda function takes each location in 'centroid', and gets its distance to the given 'location'. The built-in
    # min function then simply goes through the list of all the distances, picks the shortest distace, and returns the
    # point to which that distance corresponds.

    # Note to self: https://docs.python.org/2/library/functions.html#min -eh

def group_by_first(pairs):
    """Return a list of pairs that relates each unique key in [key, value]
    pairs to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_first(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    return [[y for x, y in pairs if x == key] for key in keys]




def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in `centroids`. Each item in
    `restaurants` should appear once in the result, along with the other
    restaurants closest to the same centroid.
    """
    # BEGIN Question 4
    "*** YOUR CODE HERE ***"
    lyst = [] #starting list, will add on to it
    for i in restaurants:
        lyst.append([find_closest(restaurant_location(i), centroids), i])
    return group_by_first(lyst)

    # Starts with an empty list. Goes through every restaurant, gets its location, and then uses find_closest to group
    # locations intro centroids. Group_by_first uses key value pairs to sort the locations by centroid


    # END Question 4


def find_centroid(cluster):
    """Return the centroid of the `cluster` based on the locations of the
    restaurants."""
    # BEGIN Question 5
    "*** YOUR CODE HERE ***"
    def get_lats(cluster):
        return [restaurant_location(item)[0] for item in cluster]

    def get_longs(cluster):
        return [restaurant_location(item)[1] for item in cluster]


    # Goes through list of all restaurants in a cluster, and seperates them into all the latitudes, and all the longitutes
    # Then finds the average lat, and average long

    avg_lat = mean(get_lats(cluster))
    avg_long = mean(get_longs(cluster))

    return [avg_lat, avg_long]
    # These are the co-ords of a centroid's cluster

    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    # if theres too few restaurants it's not possible to create a cluster

    old_centroids, n = [], 0
    # starting centroids come from a random # of restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]
    # sample gets a random # of restaurants

    while old_centroids != centroids and n < max_updates: #keeps the number of iterations under 100, makes sure the old and new centroids differ
        old_centroids = centroids
        # BEGIN Question 6

        centroids = []
        for centroid in group_by_centroid(restaurants, old_centroids):
            centroids.append(find_centroid(centroid))

            # goes through and groups the restaurants using the old centroid values. Goes and creates new centroid values, using the find_centroid function
            ## NOTE: I struggled a long time with this problem, and had to have a friend who goes to Cal walk me through it -eh

        # END Question 6
        n += 1
        #iterates the number of times this function has run
    return centroids


def find_predictor(user, restaurants, feature_fn):
    """Return a rating predictor (a function from restaurants to ratings),
    for USER by performing least-squares linear regression using FEATURE_FN
    on the items in RESTAURANTS. Also, return the R^2 value of this model.
    Arguments:
    user -- A user
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    reviews_by_user = {review_restaurant_name(review): review_rating(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]

    "*** YOUR CODE HERE ***"
    b, a, r_squared = 0, 0, 0  # REPLACE THIS LINE WITH YOUR SOLUTION, done below
    sxx, syy, sxy = 0, 0, 0
    for x in xs:
        sxx += (x - mean(xs)) * (x - mean(xs))
    for y in ys:
        syy += (y - mean(ys)) * (y - mean(ys))
    for i in range(len(xs)):
        sxy += (xs[i] - mean(xs)) * (ys[i] - mean(ys))
    b = sxy / sxx
    a = mean(ys) - b * mean(xs)
    r_squared = (sxy * sxy) / (sxx * syy)

    # These mathematical procedures were provided by the project website, there's no way we would have come up with this
    # on our own

    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within FEATURE_FNS that gives the highest R^2 value
    for predicting ratings by the user; return a predictor using that feature.
    Arguments:
    user -- A user
    restaurants -- A dictionary from restaurant names to restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = list(user_reviewed_restaurants(user, restaurants).values())
    "*** YOUR CODE HERE ***"
    result_li = []
    for feature_fn in feature_fns:
        result_li.append(find_predictor(user, reviewed, feature_fn))
    return max(result_li, key=lambda result: result[1])[0]
    # Uses sample-code builtin functions to 'predict' ratings, using the R2 value from user's previous ratings


def rate_all(user, restaurants, feature_functions):
    """Return the predicted ratings of RESTAURANTS by USER using the best
    predictor based a function from FEATURE_FUNCTIONS.
    Arguments:
    user -- A user
    restaurants -- A dictionary from restaurant names to restaurants
    """
    # Use the best predictor for the user, learned from *all* restaurants
    # (Note: the name RESTAURANTS is bound to a dictionary of all restaurants)
    predictor = best_predictor(user, restaurants, feature_functions)
    "*** YOUR CODE HERE ***"
    reviewed = user_reviewed_restaurants(user, restaurants)
    result = {}
    for r_name, r in restaurants.items():
        if r_name in reviewed.keys():
            result[r_name] = review_rating(user_reviews(user)[r_name])
        else:
            result[r_name] = predictor(r)
    return result
    # creates a dictionary with the ratings predicted from the user for each restaurant


def search(query, restaurants):
    """Return each restaurant in RESTAURANTS that has QUERY as a category.
    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    "*** YOUR CODE HERE ***"
    result = []
    for i in range(len(restaurants) - 1, -1, -1):
        if query not in restaurant_categories(restaurants[i]):
            restaurants.remove(restaurants[i])
    return restaurants
    # Goes through list of each restaurant, and sees if the user has queried it



def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_rating,
            restaurant_price,
            restaurant_num_ratings,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]

@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(CATEGORIES, 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict ratings for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_rating(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
