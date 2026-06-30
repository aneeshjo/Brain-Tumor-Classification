import sys

from cnnClassifier.exception import CustomException

try:
    10 / 0
except Exception as e:
    raise CustomException(e, sys)
