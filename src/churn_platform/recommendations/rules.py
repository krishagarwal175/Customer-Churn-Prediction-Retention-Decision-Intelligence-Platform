"""Hybrid retention recommendation rules.

Combines a per-persona playbook (segment strategy) with driver-triggered
actions (individual precision) to produce a de-duplicated, ordered set of
retention actions for a customer. The action catalog is configuration-driven so
offers can be edited without code changes.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

LOGGER = logging.getLogger(__name__)

__all__ = ["RetentionRuleEngine"]


class RetentionRuleEngine:
    """Generate retention actions from persona playbooks and churn drivers."""

    def __init__(
        self,
        personas: Mapping[str, Sequence[str]],
        driver_rules: Sequence[Mapping[str, str]],
    ) -> None:
        """Initialize the rule engine.

        Args:
            personas: Mapping of persona name to its playbook actions.
            driver_rules: Sequence of ``{"match": <substring>, "action": <text>}``
                rules; a rule fires when a readable driver contains ``match``.

        Raises:
            ValueError: If any driver rule is missing ``match`` or ``action``.
        """
        self._personas = {name: list(actions) for name, actions in personas.items()}
        self._driver_rules: list[tuple[str, str]] = []
        for rule in driver_rules:
            if "match" not in rule or "action" not in rule:
                raise ValueError("Each driver rule needs 'match' and 'action' keys.")
            self._driver_rules.append((str(rule["match"]), str(rule["action"])))
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RetentionRuleEngine:
        """Build an engine from a ``recommendations`` config mapping.

        Args:
            config: Full config dict, or the ``recommendations`` sub-mapping.

        Returns:
            A configured ``RetentionRuleEngine``.
        """
        section = config.get("recommendations", config)
        return cls(
            personas=section.get("personas", {}),
            driver_rules=section.get("driver_rules", []),
        )

    def playbook_actions(self, persona: str) -> list[str]:
        """Return the fixed playbook actions for a persona (empty if unknown)."""
        return list(self._personas.get(persona, []))

    def driver_actions(self, drivers: Sequence[str]) -> list[str]:
        """Return actions triggered by the given readable churn drivers.

        Args:
            drivers: Readable driver strings (e.g. ``"Contract = Month-to-month"``).

        Returns:
            Ordered, de-duplicated actions whose rule matched a driver.
        """
        actions: list[str] = []
        for driver in drivers:
            for match, action in self._driver_rules:
                if match in driver and action not in actions:
                    actions.append(action)
        return actions

    def recommend(self, persona: str, drivers: Sequence[str]) -> list[str]:
        """Return the hybrid recommendation: playbook first, then driver actions.

        Args:
            persona: Customer persona name.
            drivers: Readable churn drivers for the customer.

        Returns:
            Ordered, de-duplicated list of recommended retention actions.
        """
        recommended: list[str] = []
        for action in [*self.playbook_actions(persona), *self.driver_actions(drivers)]:
            if action not in recommended:
                recommended.append(action)
        return recommended
