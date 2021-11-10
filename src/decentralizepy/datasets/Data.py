class Data:
    """
    This class defines the API for Data.
    """

    def __init__(self, x, y):
        """
        Constructor
        Parameters
        ----------
        x : numpy array
            A numpy array of data samples
        y : numpy array
            A numpy array of outputs corresponding to the sample
        """
        self.x = x
        self.y = y

    def __getitem__(self, i):
        """
        Function to get the item with index i.
        Parameters
        ----------
        i : int
            Index
        Returns
        -------
        2-tuple
            A tuple of the ith data sample and it's corresponding label
        """
        return self.x[i], self.y[i]
