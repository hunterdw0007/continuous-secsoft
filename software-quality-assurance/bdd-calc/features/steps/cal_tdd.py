import unittest
import calculator

class TestCalc(unittest.TestCase):

    def testAdd(self):
        self.assertEqual(3+2, calculator.add(3, 2))
    
    def testAddError(self):
        self.assertEqual('Error: Invalid Input', calculator.add('three', '03'))

    def testSub(self):
        self.assertEqual(3-2, calculator.sub(3, 2))

    def testSubError(self):
        self.assertEqual('Error: Invalid Input', calculator.sub('3','two'))

    def testMult(self):
        self.assertEqual(4*4, calculator.mult(4, 4))

    def testMultError(self):
        self.assertEqual('Error: Invalid Input', calculator.mult('four', '0x4'))

    def testDiv(self):
        self.assertEqual(4/2, calculator.div(4, 2))
    
    def testDivZero(self):
        self.assertEqual('Error: Division By Zero', calculator.div(4, 0))

    def testDivError(self):
        self.assertEqual('Error: Invalid Input', calculator.div('four', b'08'))

if __name__ == '__main__':
    unittest.main()