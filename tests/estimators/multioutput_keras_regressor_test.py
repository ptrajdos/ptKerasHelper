import unittest

from pt_keras_helper.estimators.multioutput_keras_regressor import (
    MultiOutputKerasRegressor,
    make_model_default,
)
from sklearn.utils.estimator_checks import estimator_checks_generator

#TODO this even fails with original Keras Regressor
SKIP_CHECKS = {
    "check_estimators_data_not_an_array",
}

class MultioutputKerasRegressorTest(unittest.TestCase):

    def get_estimators(self) -> dict:
        return {
            "default": MultiOutputKerasRegressor(model=make_model_default)
            }

    def test_sklearn(self):
        for clf_name, clf in self.get_estimators().items():
            with self.subTest(clf_name=clf_name):

                for estimator, check in estimator_checks_generator(clf):
                    check_name = check.func.__name__

                    print(f"Trying to perform {check_name}")
                    if check_name in SKIP_CHECKS:
                        continue
                    
                    print(f"Checking: {check_name}")
                    check(estimator)

if __name__ == "__main__":
    unittest.main()