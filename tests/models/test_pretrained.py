import unittest

from numpy import float32

from ecnet.models.pretrained import predict


class TestPretrained(unittest.TestCase):

    def test_pretrained(self):

        print('\nUNIT TEST: Pretrained Models')
        props = [
            ('CN', 'padel'),
            ('CP', 'padel'),
            ('KV', 'padel'),
            ('PP', 'padel'),
            ('YSI', 'padel')
        ]
        for p in props:
            pred, err = predict(['CCCCC'], p[0], p[1])
            print(p[0], p[1], pred[0][0], err)
            self.assertEqual(type(err), float)
            self.assertEqual(type(pred[0][0]), float32)


if __name__ == '__main__':

    unittest.main()
