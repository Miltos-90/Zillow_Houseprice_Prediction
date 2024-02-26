import sys
from typing import List
from pyspark.ml.tuning import ParamGridBuilder


class RandomGridBuilder(ParamGridBuilder): 
  """ 
  Grid builder for random search. Sets up grids for use in CrossValidator in Spark using values randomly sampled from user-provided distributions.
  Distributions should be provided as lambda functions, so that the numbers are generated at call time. 
  It is based on the Spark ML ParamGridBuilder class of pyspark [1] and can be used in its place.
  
  Args
    numIterations: int
      Number of parameter settings that are sampled.
    seed: int (optional)
      Seed for random sampling. Set to None if you're setting the seed
      outside of this class.
    
  Outputs
    param_map: list of parameter maps to use in cross validation (CrossValidator class pf pyspark).

  
  References
    [1] https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/tuning.html#ParamGridBuilder
  """
  
  def __init__(self, numIterations: int, seed: int = 42):
    self._param_grid = dict()
    self.nIter = numIterations
    self.seed  = seed

    return
  

  def _setSeed(self, iterNum: int) -> None:
    """ Sets the seed """

    # Set seeds if needed
    if self.seed:

      if 'numpy' in sys.modules:
        # We don't know the set name of the imported module (probably np in this case), 
        # but avoid using (np.seed(...)) directly as it might break. 
        from numpy.random import seed as __npseed
        __npseed(self.seed + iterNum)

      if 'random' in sys.modules:
        # Same as above
        from random import seed as __randseed
        __randseed(self.seed + iterNum)

      return
  

  def build(self) -> List:
    """
      Builds and returns all parameter combinations specified by the parameter grid.
    """

    parameterMap = []

    for n in range(self.nIter):

      self._setSeed(n)
      
      paramDict = { # name: value (drawn from the user's corresponding distribution) pairs
        parameterName: randomDistFunc() 
        for parameterName, randomDistFunc in self._param_grid.items() 
      }

      parameterMap.append(paramDict)
    
    return parameterMap
  

if __name__ == "__main__":

  from pyspark.ml.classification import LogisticRegression
  import numpy as np 

  lr = LogisticRegression()

  paramGrid = (
    RandomGridBuilder(3)
    .addGrid(lr.regParam, lambda: np.random.rand())
    .addGrid(lr.maxIter,  lambda: np.random.randint(10))
    .build()
  )

  for parameters in paramGrid:
    print(parameters)