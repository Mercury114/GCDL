"""Trustworthy vision-text emotion conflict routing."""

from .routing import RouteDecision, choose_dual_channel_route, choose_route

__all__ = ["RouteDecision", "choose_dual_channel_route", "choose_route"]
