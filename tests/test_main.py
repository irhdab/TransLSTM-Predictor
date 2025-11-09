
import unittest
from main import main

class TestMain(unittest.TestCase):

    def test_main_runs_without_errors(self):
        """
        Test that the main function runs without errors.
        """
        try:
            main()
        except Exception as e:
            self.fail(f'main() raised {e} unexpectedly!')

if __name__ == '__main__':
    unittest.main()
