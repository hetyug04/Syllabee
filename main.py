import os
import asyncio
import uvicorn
from bot import bot
from planner import app as api

import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

async def main():
    # Start the FastAPI server in a separate task
    api_config = uvicorn.Config(api, host="0.0.0.0", port=8000, log_level="info")
    api_server = uvicorn.Server(api_config)
    api_task = asyncio.create_task(api_server.serve())

    # Start the Discord bot
    bot_token = os.getenv("DISCORD_TOKEN")
    if not bot_token:
        logger.error("Error: DISCORD_TOKEN environment variable not set.")
        return
    bot_task = asyncio.create_task(bot.start(bot_token))

    # Wait for both tasks to complete (they will run indefinitely)
    await asyncio.gather(api_task, bot_task)

if __name__ == "__main__":
    # Ensure DISCORD_TOKEN is set for local testing
    # os.environ["DISCORD_TOKEN"] = "YOUR_DISCORD_BOT_TOKEN_HERE" # Uncomment and replace for local testing

    asyncio.run(main())