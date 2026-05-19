#!/usr/bin/env python3
"""
Export SBODEMOUS (SAP B1 demo data schema) metadata + non-empty tables to CSV.gz on disk.
Run on a host that has hdbcli and network access to HANA.
Credentials: prefer env HANA_PASSWORD; else optional CLI arg (avoid logging).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import os
import sys
from typing import Any

from hdbcli import dbapi


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("HANA_HOST", "hana.sg.sapb1c.com"))
    p.add_argument("--port", type=int, default=int(os.environ.get("HANA_PORT", "30015")))
    p.add_argument("--user", default=os.environ.get("HANA_USER", "AIRO01"))
    p.add_argument("--password", default=os.environ.get("HANA_PASSWORD"))
    p.add_argument("--schema", default=os.environ.get("HANA_SCHEMA", "SBODEMOUS"))
    p.add_argument(
        "--out",
        default=None,
        help="Backup root directory (default: ~/hana-backups/dataschema_<SCHEMA>_<timestamp>)",
    )
    args = p.parse_args()
    if not args.password:
        print("Missing password: set HANA_PASSWORD or pass --password", file=sys.stderr)
        return 2

    schema = args.schema.upper()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.out or os.path.expanduser(f"~/hana-backups/dataschema_{schema}_{ts}")
    data_dir = os.path.join(backup_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    conn = dbapi.connect(args.host, args.port, args.user, args.password)
    cur = conn.cursor()
    cur.execute(f"SET SCHEMA {qident(schema)}")

    cur.execute(
        f"""
        SELECT TABLE_NAME, RECORD_COUNT, TABLE_SIZE, TABLE_TYPE, IS_COLUMN_TABLE
        FROM M_TABLES
        WHERE SCHEMA_NAME = '{schema}'
        ORDER BY TABLE_NAME
        """
    )
    manifest_rows = cur.fetchall()
    mf = os.path.join(backup_root, "manifest_m_tables.csv")
    with open(mf, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["TABLE_NAME", "RECORD_COUNT", "TABLE_SIZE", "TABLE_TYPE", "IS_COLUMN_TABLE"]
        )
        w.writerows(manifest_rows)

    readme = os.path.join(backup_root, "README.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "HANA data schema backup (manifest + non-empty tables as CSV.gz)\n"
            f"Time (UTC server local): {ts}\n"
            f"HANA host: {args.host}\n"
            f"Port: {args.port}\n"
            f"User: {args.user}\n"
            f"Schema: {schema}\n"
            "Password is NOT stored in this folder.\n"
        )

    err_path = os.path.join(backup_root, "export_errors.log")
    exported = 0
    skipped_empty = 0
    with open(err_path, "w", encoding="utf-8") as errlog:
        for table_name, record_count, *_rest in manifest_rows:
            rc = record_count if record_count is not None else 0
            if rc == 0:
                skipped_empty += 1
                continue
            safe = table_name.replace("/", "_").replace("\\", "_")
            out_gz = os.path.join(data_dir, f"{safe}.csv.gz")
            sql = f"SELECT * FROM {qident(schema)}.{qident(table_name)}"
            try:
                cur.execute(sql)
                colnames = [d[0] for d in cur.description]
                with gzip.open(out_gz, "wt", encoding="utf-8", newline="") as gf:
                    wr = csv.writer(gf)
                    wr.writerow(colnames)
                    while True:
                        batch: list[Any] = cur.fetchmany(8000)
                        if not batch:
                            break
                        wr.writerows(batch)
                exported += 1
            except Exception as e:
                errlog.write(f"{table_name}: {type(e).__name__}: {e}\n")

    cur.close()
    conn.close()

    summary = os.path.join(backup_root, "SUMMARY.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(
            f"backup_root={backup_root}\n"
            f"tables_in_manifest={len(manifest_rows)}\n"
            f"skipped_empty={skipped_empty}\n"
            f"exported_gzip={exported}\n"
        )
    print(summary)
    with open(summary, encoding="utf-8") as f:
        print(f.read().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
