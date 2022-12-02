class Recommender():
    """Recommender class, to be subclassed."""
    def __init__(self, datasets: list[str]):
        pass

    def recommend(self, ratings: dict) -> list[str]:
        """ratings should be a mapping of movie to rating by a user"""
        pass