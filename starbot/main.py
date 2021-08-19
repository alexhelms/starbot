import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging
from discord.ext import commands
from starbot.cogs.analysis import AnalysisCog

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    TOKEN = os.getenv('DISCORD_TOKEN')
    if TOKEN is None:
        print('DISCORD_TOKEN is empty', file=sys.stderr)
        sys.exit(1)

    bot = commands.Bot(command_prefix=commands.when_mentioned)
    bot.add_cog(AnalysisCog(bot))
    bot.run(TOKEN)
