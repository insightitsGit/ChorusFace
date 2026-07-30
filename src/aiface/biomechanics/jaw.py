"""Second-order jaw physics. Speech sets a target; physics settles the angle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class JawState:
    angle: float = 0.0
    velocity: float = 0.0
    angular_momentum: float = 0.0
    target: float = 0.0


@dataclass(slots=True)
class JawSystem:
    # Heavier jaw so openness changes over ~200 ms instead of popping.
    mass: float = 1.25
    damping: float = 6.2
    elasticity: float = 22.0
    max_opening: float = 0.62
    state: JawState | None = None

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = JawState()

    @property
    def angle(self) -> float:
        assert self.state is not None
        return self.state.angle

    def set_speech_target(self, openness_norm: float) -> None:
        """Map [0,1] desired mouth openness into a jaw angle target."""
        assert self.state is not None
        self.state.target = max(0.0, min(1.0, openness_norm)) * self.max_opening

    def step(self, dt: float) -> float:
        assert self.state is not None
        if dt <= 0.0:
            return self.state.angle
        # Spring toward target + elastic return toward rest (0).
        torque = (self.state.target - self.state.angle) * self.elasticity
        torque -= self.state.angle * (self.elasticity * 0.35)
        torque -= self.state.velocity * self.damping
        acceleration = torque / max(self.mass, 1e-3)
        self.state.velocity += acceleration * dt
        self.state.angular_momentum = self.state.velocity * self.mass
        self.state.angle += self.state.velocity * dt
        self.state.angle = max(0.0, min(self.max_opening, self.state.angle))
        if self.state.angle in (0.0, self.max_opening):
            self.state.velocity *= -0.15
        return self.state.angle

    def openness_units(self, *, rest: float = 1.0, scale: float = 18.0) -> float:
        """Convert radians into the render-space mouth openness units."""
        return rest + (self.angle / max(self.max_opening, 1e-6)) * scale


__all__ = ["JawState", "JawSystem"]
