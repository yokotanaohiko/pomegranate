#!/usr/bin/env python
# -*- coding:utf-8 -*-

from random import random

class Buggage:
    def __init__(self, damage=None):
        if damage is not None:
            self.damage = damage
        else:
            self.damage = 5

    def fixed(self):
        return self.damage <= 0

class StreamBase:
    def __init__(self):
        self.children = []
        self.persons = []
        self.queue = []

    def tick(self):
        for child in self.children:
            child.tick()

    def subscribe(self, stream):
        if len(self.children) < 1:
            self.children.append(stream)
        else:
            self.children[0] = stream
        return stream

    def has_child(self):
        return len(self.children) > 0

    def add_person(self, person):
        self.persons.append(person)

    def move_person(self):
        if len(self.persons) > 0:
            self.persons.pop()

    def __str__(self):
        return self.__class__.__name__

class SingleStream(StreamBase):
    def __init__(self):
        super().__init__()


class DoubleStream(StreamBase):
    def __init__(self):
        super().__init__()

    def subscribe(self, stream1, stream2):
        if len(self.children) < 1:
            self.children.append(stream1)
        else:
            self.children[0] = stream1

        if len(self.children) < 2:
            self.children.append(stream2)
        else:
            self.children[1] = stream2

        return stream1
   
    def get_first_stream(self):
        try:
            return self.children[0]
        except IndexError:
            raise IndexError('一つ目の子要素が存在しません')

    def get_second_stream(self):
        try:
            return self.children[1]
        except IndexError:
            raise IndexError('二つ目の子要素が存在しません')

class Person:
    def __init__(self):
        pass

    def fix(self, buggage):
        buggage.damage -= 1

class Line:
    def __init__(self, root_stream=None):
        if root_stream is None:
            self.root_stream = SingleStream()
        else:
            self.root_stream = root_stream

    def get_leaf_streams(self):
        leaf_stream_list = []
        node_queue = [self.root_stream]
        while len(node_queue) > 0:
            next_node = node_queue.pop(0)
            for child in next_node.children:
                if child.has_child():
                    node_queue.append(child)
                else:
                    leaf_stream_list.append(child)
        return leaf_stream_list
    
    def tick(self):
        self.root_stream.tick()

    def __str__(self):
        result = ''
        node_queue = [self.root_stream]
        while len(node_queue) > 0:
            next_node = node_queue.pop(0)
            result += str(next_node) + '\n'
            for i, child in enumerate(next_node.children):
                result += '|' if i == 0 else '----|'
                if child.has_child():
                    node_queue.append(child)
                else:
                    
            result += '\n'
        return result



class FixableDetectionStream(DoubleStream):

    BROKEN_PROBABILITY = 0.2

    def tick(self):
        for person in self.persons:
            if len(self.queue) <= 0:
                break

            buggage = self.queue.pop()
            if random() < self.BROKEN_PROBABILITY:
                self.get_second_stream().queue.append(buggage)
            else:
                self.get_first_stream().queue.append(buggage)

        super().tick()

class FixStream(SingleStream):
    def tick(self):
        self._fix()
        super().tick()

    def _fix(self):
        for person in self.persons:
            if len(self.queue) <= 0:
                break
            
            buggage = self.queue[0]
            person.fix(buggage)

            if buggage.fixed():
                try:
                    self.children[0].queue.append(self.queue.pop(0))
                except IndexError:
                    raise IndexError('一つ目の子要素が存在しません')

class DiscardStream(SingleStream):
    pass

line = Line()
(line.root_stream
    .subscribe(FixableDetectionStream())
    .subscribe(FixStream(), DiscardStream())
    .subscribe(SingleStream())
)

print(line.get_leaf_streams())
line.tick()
print(line)
