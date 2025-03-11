import unittest
from lib.zipper import zip_string, unzip_string

class TestZipper(unittest.TestCase):
    def test_zip_string(self):
        input_string = "This is a test string"
        compressed_string = zip_string(input_string)
        self.assertTrue(isinstance(compressed_string, str))
        self.assertTrue(len(compressed_string) > 0)
    
    def test_unzip_string(self):
        input_string = "This is a test string"
        compressed_string = zip_string(input_string)
        uncompressed_string = unzip_string(compressed_string)
        self.assertTrue(isinstance(uncompressed_string, str))
        self.assertEqual(uncompressed_string, input_string)

    def test_zipped_string_is_compressed(self):
        input_string = "This is a test string very long with many characters. Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        compressed_string = zip_string(input_string)
        self.assertTrue(len(compressed_string) < len(input_string))

if __name__ == '__main__':
    unittest.main()