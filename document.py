from random import sample

class Document(object):

    def __init__(self, name):
        self.pos = Positive()
        self.neg = set()
        self.name = None
    
    def add_to_pos(self, i, cell, span):
        if i in cell:
            self.pos.add_span(i)
        elif:
            self.pos.add_cell(i)
        else:
            assert False
    
    def add_to_neg(self, i):
        self.neg.add(i)
    
    def get_sample(self):
        span_id = sample(self.pos.span, 1)
        cell_id = sample(self.pos.cell, 1)
        pos = [span_id, cell_id]

        num_neg = 2 * int(len(self.neg)/len(self.pos))

        return set(pos), set(sample(self.neg, num_neg))


class Positive(object):
    def __init__(self):
        self.span = set()
        self.cell = set()

    def add_span(self, i):
        self.span.add(i)

    def add_cell(self, i):
        self.cell.add(i)

    def __len__(self):
        return len(self.span)+len(self.cell)
