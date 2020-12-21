from sklearn.ensemble import RandomForestClassifier


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return RandomForestClassifier(random_state=5).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)