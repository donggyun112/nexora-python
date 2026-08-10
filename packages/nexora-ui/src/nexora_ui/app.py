"""ASGI composition root for the optional Nexora test UI."""

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from .api import router
from .config import SETTINGS


def create_app() -> FastAPI:
    """Create and configure the test console application."""
    application = FastAPI(title="Nexora Durable Agent Lab")

    @application.middleware("http")
    async def disable_ui_cache(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Disable browser caching for the UI shell and static assets."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    application.include_router(router)
    application.mount(
        "/assets",
        StaticFiles(directory=SETTINGS.ui_root / "static"),
        name="assets",
    )

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """Serve the test console entry page."""
        return FileResponse(SETTINGS.ui_root / "static" / "index.html")

    return application


app = create_app()
