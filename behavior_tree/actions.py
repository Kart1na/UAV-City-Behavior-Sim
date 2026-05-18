"""Behavior-tree action nodes."""

from __future__ import annotations

from .core import Node, Status


class IncreasePanic(Node):
    def __init__(self, amount: float = 0.2):
        self.amount = amount

    def tick(self, agent, world) -> Status:
        agent.panic = min(1.0, agent.panic + self.amount)
        return Status.SUCCESS


class RunAwayFromEvent(Node):
    def tick(self, agent, world) -> Status:
        event = world.get("events").nearest_active_event(agent.position) if world.get("events") else None
        if not event:
            return Status.FAILURE
        agent.run_away_from(event.position)
        return Status.SUCCESS


class FollowCrowd(Node):
    def tick(self, agent, world) -> Status:
        humans = world.get("humans")
        if not humans:
            return Status.FAILURE
        runners = [human for human in humans.humans if human.state == "running"]
        if not runners:
            return Status.FAILURE
        avg_x = sum(human.goal[0] for human in runners) / len(runners)
        avg_y = sum(human.goal[1] for human in runners) / len(runners)
        agent.set_goal((avg_x, avg_y))
        return Status.SUCCESS


class WalkToGoal(Node):
    def tick(self, agent, world) -> Status:
        agent.state = "walking" if agent.panic <= 0.55 else "running"
        return Status.SUCCESS


class AvoidObstacles(Node):
    def tick(self, agent, world) -> Status:
        scene = world.get("scene")
        if not scene:
            return Status.SUCCESS
        nearby = scene.get_obstacles_in_radius(agent.position, radius=4.0)
        if not nearby:
            return Status.SUCCESS
        ox, oy = nearby[0].center()
        agent.set_goal((agent.position[0] - (ox - agent.position[0]), agent.position[1] - (oy - agent.position[1])))
        return Status.SUCCESS

