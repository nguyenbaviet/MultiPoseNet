from abc import abstractmethod

class A:
    def __init__(self):
        print('-----------BestMonster----------------')
    @abstractmethod
    def printer(self):
        print('-----SAY HI-------')
class B(A):
    def __init__(self):
        super().__init__()
    def printer(self):
        print('nothing')
if __name__ == '__main__':
    c = [5, 6]
    def printer(*input):
        print(input)
    printer(c)