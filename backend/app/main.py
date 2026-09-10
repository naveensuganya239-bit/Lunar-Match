from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes import router
from app.utils.logging_config import logger

app = FastAPI(
    title="Lunar Multi-Sensor Image Registration API",
    description=(
        "Registers TMC, IIRS, and SAR lunar source images to the OHRC "
        "reference image, computes real quantitative registration metrics, "
        "fuses the results into a multi-sensor product, and generates a "
        "confidence-aware Lunar AI scientific interpretation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Serve generated outputs directly as static files too (in addition to the
# /api/download endpoint), which is convenient for <img src="..."> tags.
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    logger.info("Lunar Multi-Sensor Registration API starting up.")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Frontend directory: {config.FRONTEND_DIR}")

    # Ensure demo dataset exists so the web app is immediately testable out of the box
    try:
        from app.services.dataset import find_sensor_image
        if not find_sensor_image("OHRC"):
            logger.info("No sensor images found in data directory. Auto-generating demo dataset...")
            from scripts.generate_demo_dataset import main as gen_demo
            gen_demo()
            logger.info("Demo dataset generated successfully.")
    except Exception as e:
        logger.warning(f"Could not check/generate demo dataset: {e}")


# Serve the frontend as a static site at "/" — this MUST come last so that
# the /api/* and /outputs/* routes defined above take priority.
# html=True makes "/" serve index.html automatically.
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
    logger.info(f"Serving frontend from {config.FRONTEND_DIR}")
