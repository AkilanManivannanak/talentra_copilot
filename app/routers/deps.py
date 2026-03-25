from __future__ import annotations

from fastapi import Request

from app.services.container import ServiceContainer


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services
