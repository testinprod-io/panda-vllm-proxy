from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from compaction import compact
from .log import logger

def job_wrapper():
    """Wrapper function to add logging around the compaction job"""
    try:
        logger.info("=== CRON JOB TRIGGERED ===")
        compact()
        logger.info("=== CRON JOB COMPLETED ===")
    except Exception as e:
        logger.error(f"=== CRON JOB FAILED: {str(e)} ===", exc_info=True)

sched = BlockingScheduler()
sched.add_job(
    job_wrapper,
    CronTrigger(minute=0), # every hour
    misfire_grace_time=300
)

logger.info("Starting cron scheduler - compaction will run every hour at minute 0")

try:
    sched.start()
except Exception as e:
    logger.error(f"Scheduler error: {str(e)}", exc_info=True)