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
if __name__ == '__main__':
    b = B()
    b.printer()