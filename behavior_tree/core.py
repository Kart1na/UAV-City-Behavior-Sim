"""Small behavior tree framework."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class Tickable(Protocol):
    def tick(self, agent, world) -> Status:
        ...


class Node:
    def tick(self, agent, world) -> Status:
        raise NotImplementedError


class Selector(Node):
    def __init__(self, *children: Tickable):
        self.children = list(children)

    def tick(self, agent, world) -> Status:
        for child in self.children:
            status = child.tick(agent, world)
            if status in (Status.SUCCESS, Status.RUNNING):
                return status
        return Status.FAILURE


class Sequence(Node):
    def __init__(self, *children: Tickable):
        self.children = list(children)

    def tick(self, agent, world) -> Status:
        for child in self.children:
            status = child.tick(agent, world)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

