-- A monthly archive that 404s after the symbol's listing is "not published
-- yet", not "done". Earlier code recorded it as 'success' with 0 rows, which
-- skipped that month forever. Reclassify any such rows as 'missing' so the
-- next backfill re-requests them. The backfill loop never starts before the
-- listing date, so every 0-row success row is one of these.
UPDATE ingestion_runs
   SET status = 'missing'
 WHERE component = 'backfill'
   AND status = 'success'
   AND rows_written = 0;
