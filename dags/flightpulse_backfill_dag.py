import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag


def _months(start="2019-01", end="2023-08"):
    cur = pendulum.from_format(start, "YYYY-MM")
    last = pendulum.from_format(end, "YYYY-MM")
    out = []
    while cur <= last:
        nxt = cur.add(months=1)
        out.append((cur.format("YYYY-MM-DD"),
                    nxt.subtract(days=1).format("YYYY-MM-DD")))
        cur = nxt
    return out


@dag(schedule=None, start_date=pendulum.datetime(2019, 1, 1, tz="UTC"),
     catchup=False, tags=["flightpulse", "backfill"])
def flightpulse_backfill():
    prev = None
    for i, (ws, we) in enumerate(_months()):
        t = TriggerDagRunOperator(
            task_id=f"replay_{ws[:7].replace('-', '_')}",
            trigger_dag_id="flightpulse_pipeline",
            conf={
                "reference_start": "2019-06-01", "reference_end": "2019-06-30",
                "window_start": ws, "window_end": we,
            },
            wait_for_completion=False,
        )
        if prev:
            prev >> t
        prev = t


flightpulse_backfill()
