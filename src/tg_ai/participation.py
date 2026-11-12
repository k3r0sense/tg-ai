"""Participation policy: direct replies bypass cooldown; unsolicited turns are bounded."""
import random
import time
from dataclasses import dataclass


@dataclass
class Activity:
    last_turn: float = float('-inf')
    last_check: float = float('-inf')
    humans: int = 0


class Participation:
    def __init__(self, settings, clock=time.monotonic, rng=random.random):
        self.settings, self.clock, self.rng = settings, clock, rng
        self.states = {}

    def observe(self, scope):
        self.states.setdefault(scope, Activity()).humans += 1

    def eligible(self, scope):
        state = self.states.setdefault(scope, Activity())
        now = self.clock()
        if not self.settings.proactive_enabled:
            return False
        if state.humans < self.settings.proactive_min_messages:
            return False
        if now - state.last_turn < self.settings.proactive_cooldown_seconds:
            return False
        if now - state.last_check < self.settings.decision_interval_seconds:
            return False
        state.last_check = now
        return True

    def spontaneous(self):
        return self.rng() < self.settings.spontaneous_probability

    def sent(self, scope):
        state = self.states.setdefault(scope, Activity())
        state.last_turn = self.clock()
        state.humans = 0
