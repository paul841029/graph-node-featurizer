from random import sample

class Document(object):

    def __init__(self, name):
        self.pos = Positive()
        self.neg = set()
        self.name = name
    
    def add_to_pos(self, i, cell, span):
        if i in span:
            pass
        elif i in cell:
            self.pos.add_cell(i)
        else:
            assert False
    
    def add_pos_span(self, gt_group_span):
        for d in gt_group_span:
            if d['file'].split("/")[-1] == self.name:
                self.pos.add_span(d['example'])
    
    def add_to_neg(self, i):
        self.neg.add(i)
    
    def get_sample(self):
        num = 0
        if len(self.pos.cell) > 0:
            cell_id = sample(self.pos.cell, 1)
            num += 1
        else:
            cell_id = []
        
        if len(self.pos.span) > 0:
            span_id = sample(self.pos.span, 1)[0]
            num += 1
        else:
            span_id = []
        
        pos = span_id + cell_id

        try:
            num_neg = num * 10
        except:
            num_neg = 0

        if num_neg > len(self.neg):
            num_neg = len(self.neg)

        return set(pos), set(sample(self.neg, num_neg)), num


class Positive(object):
    def __init__(self):
        self.span = []
        self.cell = []

    def add_span(self, i):
        self.span.append(i)

    def add_cell(self, i):
        self.cell.append(i)

    def __len__(self):
        return len(self.span)+len(self.cell)
