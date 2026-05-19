#!/usr/bin/env python3
"""
Rebuild a HANA schema backup into a local PostgreSQL database.

The script intentionally uses only the Python standard library plus the local
`psql` client so it can run on a workstation without installing a Python PG
driver.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


TABLE_START_PATTERN = re.compile(
    r'CREATE (?:COLUMN|ROW) TABLE\s+"SBODEMOUS"\."([^"]+)"\s*\(',
    re.S,
)
VIEW_PATTERN = re.compile(
    r'CREATE VIEW\s+"SBODEMOUS"\."([^"]+)"\s+AS\n(.*?);\n\n',
    re.S,
)
SECURITY_JOIN_PATTERN = re.compile(
    r'\n\s*(?:INNER|LEFT|RIGHT)\s+JOIN\s+'
    r'"(AgentAll_USERRIGHT|AgentAll_USERBPL)"\s*'
    r'\(\s*SESSION_CONTEXT\(\'USERCODE\'\)\s*\)\s+'
    r'(?P<alias>"?[\w]+"?)\s+ON.*?(?=\n\s*(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN|\n\s*WHERE\b|\n\s*GROUP\b|\n\s*ORDER\b|\n\s*LIMIT\b|$)',
    re.I | re.S,
)
ALIAS_FROM_JOIN_PATTERN = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:\([^)]*\)|"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s+("?[\w]+"?)',
    re.I | re.S,
)
SUBQUERY_ALIAS_PATTERN = re.compile(
    r'\)\s+("?[\w]+"?)\s*(?=(?:(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\b|WHERE\b|ON\b|GROUP\b|ORDER\b|UNION\b|LIMIT\b|$))',
    re.I | re.S,
)
FROM_JOIN_OBJECT_PATTERN = re.compile(
    r'\b(?:FROM|JOIN)\s+'
    r'(?P<object>(?:(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\.)?(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))(?!\s*\()'
    r'(?:\s+(?P<alias>"?[\w]+"?))?',
    re.I | re.S,
)
CSV_FALLBACK_TABLES = {
    "AgentAll_MSG_LANG",
    "AgentAll_PARAM",
    "AgentAll_TRANS_INFO_AI",
    "AgentAll_TRANS_LOG",
    "AgentAll_USERPORT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-tar",
        default="hana-backups/dataschema_SBODEMOUS_20260404_104607.tar.gz",
    )
    parser.add_argument(
        "--ddl-part1",
        default="hana-backups/hana-ddl-SBODEMOUS-part1-tables-views.sql",
    )
    parser.add_argument(
        "--ddl-part2",
        default="hana-backups/hana-ddl-SBODEMOUS-part2-rest.sql",
    )
    parser.add_argument(
        "--pg-dsn",
        default="postgresql:///fina_demo_local",
        help="PostgreSQL DSN for the target database.",
    )
    parser.add_argument("--target-schema", default="sbodemous")
    parser.add_argument("--scope", default="full")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_dsn(dsn: str) -> tuple[str, str]:
    if "://" not in dsn:
        return dsn, "postgres"
    parsed = urlparse(dsn)
    dbname = parsed.path.lstrip("/") or "postgres"
    admin = parsed._replace(path="/postgres")
    if not parsed.netloc:
        admin_dsn = f"{parsed.scheme}:///{'postgres'}"
    else:
        admin_dsn = urlunparse(admin)
    return admin_dsn, dbname


def run_cmd(
    cmd: list[str],
    *,
    input_bytes: bytes | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.PIPE if input_bytes is not None else None,
        "stdout": subprocess.PIPE if capture else None,
        "stderr": subprocess.PIPE if capture else None,
        "check": check,
    }
    return subprocess.run(cmd, input=input_bytes, **kwargs)


def psql_exec(
    dsn: str,
    sql: str,
    *,
    capture: bool = False,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    cmd = ["psql", dsn, "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", sql]
    return run_cmd(cmd, input_bytes=input_bytes, capture=capture)


def psql_query_scalar(dsn: str, sql: str) -> str:
    cmd = ["psql", dsn, "-X", "-A", "-t", "-q", "-v", "ON_ERROR_STOP=1", "-c", sql]
    proc = run_cmd(cmd, capture=True)
    return proc.stdout.decode("utf-8").strip()


def ensure_database(target_dsn: str) -> tuple[str, str]:
    admin_dsn, dbname = parse_dsn(target_dsn)
    exists = psql_query_scalar(
        admin_dsn,
        f"SELECT 1 FROM pg_database WHERE datname = {quote_literal(dbname)};",
    )
    if exists != "1":
        psql_exec(admin_dsn, f"CREATE DATABASE {qident(dbname)};")
    return admin_dsn, dbname


def recreate_database(admin_dsn: str, dbname: str) -> None:
    psql_exec(
        admin_dsn,
        "SELECT pg_terminate_backend(pid) "
        "FROM pg_stat_activity "
        f"WHERE datname = {quote_literal(dbname)} "
        "AND pid <> pg_backend_pid();",
    )
    exists = psql_query_scalar(
        admin_dsn,
        f"SELECT 1 FROM pg_database WHERE datname = {quote_literal(dbname)};",
    )
    if exists == "1":
        psql_exec(admin_dsn, f"DROP DATABASE {qident(dbname)};")
    psql_exec(admin_dsn, f"CREATE DATABASE {qident(dbname)};")


def create_schema(target_dsn: str, schema: str) -> None:
    psql_exec(target_dsn, f"CREATE SCHEMA {qident(schema)};")


def load_manifest(tf: tarfile.TarFile) -> list[dict[str, str]]:
    member = next(m for m in tf.getmembers() if m.name.endswith("/manifest_m_tables.csv"))
    with tf.extractfile(member) as fh:
        assert fh is not None
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))


def load_csv_members(tf: tarfile.TarFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for member in tf.getmembers():
        if member.isfile() and member.name.endswith(".csv.gz") and "/data/" in member.name:
            table_name = Path(member.name).name[:-7]
            members[table_name] = member.name
    return members


def parse_table_definitions(sql_text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for match in TABLE_START_PATTERN.finditer(sql_text):
        table_name = match.group(1)
        idx = match.end()
        depth = 1
        while idx < len(sql_text) and depth > 0:
            char = sql_text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            idx += 1
        if depth == 0:
            definitions[table_name] = sql_text[match.end() : idx - 1]
    return definitions


def parse_view_definitions(sql_text: str) -> dict[str, str]:
    return {name: body for name, body in VIEW_PATTERN.findall(sql_text)}


def convert_table_body(body: str) -> str:
    converted = body
    replacements = [
        (r"\bLONGDATE\s+CS_LONGDATE\b", "timestamp without time zone"),
        (r"\bDAYDATE\s+CS_DAYDATE\b", "date"),
        (r"\bDAYDATE\b", "date"),
        (r"\bSECONDDATE\b", "timestamp without time zone"),
        (r"\bTINYINT\s+CS_INT\b", "smallint"),
        (r"\bINTEGER\s+CS_INT\b", "integer"),
        (r"\bSMALLINT\s+CS_INT\b", "smallint"),
        (r"\bBIGINT\s+CS_INT\b", "bigint"),
        (r"\bDECIMAL\((\d+\s*,\s*\d+)\)\s+CS_FIXED\b", r"numeric(\1)"),
        (r"\bSMALLDECIMAL\((\d+\s*,\s*\d+)\)\b", r"numeric(\1)"),
        (r"\bDOUBLE\b", "double precision"),
        (
            r"\bTIMESTAMP\b(?!\s+WITHOUT\s+TIME\s+ZONE)",
            "timestamp without time zone",
        ),
        (r"\bNVARCHAR\((\d+)\)", r"varchar(\1)"),
        (r"\bNVARCHAR\b(?!\s*\()", "text"),
        (r"\bNCHAR\((\d+)\)", r"char(\1)"),
        (r"\bALPHANUM\((\d+)\)", r"varchar(\1)"),
        (r"\bSHORTTEXT\((\d+)\)", r"varchar(\1)"),
        (r"\bBINTEXT\b", "text"),
        (r"\bNCLOB(?:\s+MEMORY THRESHOLD\s+\d+)?\b", "text"),
        (r"\bCLOB(?:\s+MEMORY THRESHOLD\s+\d+)?\b", "text"),
        (r"\bBLOB(?:\s+MEMORY THRESHOLD\s+\d+)?\b", "bytea"),
    ]
    for pattern, replacement in replacements:
        converted = re.sub(pattern, replacement, converted, flags=re.I)
    converted = re.sub(r"\bUNIQUE\s+BTREE\b", "UNIQUE", converted, flags=re.I)
    converted = re.sub(r"\bPRIMARY\s+KEY\s+BTREE\b", "PRIMARY KEY", converted, flags=re.I)
    converted = re.sub(r"\s+CS_[A-Z_]+\b", "", converted)
    converted = re.sub(r"\s{2,}", " ", converted)
    return converted.strip()


def build_table_sql(schema: str, table_name: str, body: str) -> str:
    converted_body = convert_table_body(body)
    return f"CREATE TABLE {qident(schema)}.{qident(table_name)} ({converted_body});"


def load_csv_header(tf: tarfile.TarFile, member_name: str) -> list[str]:
    with tf.extractfile(member_name) as raw:
        assert raw is not None
        with gzip.open(raw, "rt", encoding="utf-8", newline="") as gz:
            reader = csv.reader(gz)
            return next(reader)


def build_fallback_table_sql(schema: str, table_name: str, columns: list[str]) -> str:
    cols = ", ".join(f"{qident(col)} text" for col in columns)
    return f"CREATE TABLE {qident(schema)}.{qident(table_name)} ({cols});"


def convert_days_between(sql: str) -> str:
    pattern = re.compile(r"DAYS_BETWEEN\s*\(\s*([^(),]+?)\s*,\s*([^()]+?)\s*\)", re.I)
    previous = None
    current = sql
    while previous != current:
        previous = current
        current = pattern.sub(r"((\1)::date - (\2)::date)", current)
    return current


def strip_security_joins(sql: str) -> str:
    aliases: list[str] = []

    def repl(match: re.Match[str]) -> str:
        aliases.append(match.group("alias").replace('"', ""))
        return ""

    stripped = SECURITY_JOIN_PATTERN.sub(repl, sql)
    stripped = re.sub(r"SESSION_CONTEXT\('USERCODE'\)", "'LOCAL_DEMO'", stripped, flags=re.I)
    for alias in aliases:
        stripped = re.sub(rf"\s+AND\s+{re.escape(alias)}\.[^\n]*", "", stripped, flags=re.I)
        stripped = re.sub(rf"\s+OR\s+{re.escape(alias)}\.[^\n]*", "", stripped, flags=re.I)
        stripped = re.sub(rf"\bWHERE\s+{re.escape(alias)}\.[^\n]*\n", "WHERE\n", stripped, flags=re.I)
    stripped = re.sub(r"\bWHERE\s*(?:\n\s*)?(?=GROUP\b|ORDER\b|LIMIT\b|$)", "", stripped, flags=re.I)
    return stripped


def normalize_quoted_alias_references(sql: str) -> str:
    reference_map: dict[str, str] = {}
    keywords = {"ON", "WHERE", "GROUP", "ORDER", "LIMIT", "UNION", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "JOIN"}
    for match in FROM_JOIN_OBJECT_PATTERN.finditer(sql):
        object_token = match.group("object")
        alias_token = match.group("alias")
        if alias_token:
            alias_name = alias_token.strip('"')
            if alias_name.upper() not in keywords:
                reference_map[alias_name.upper()] = alias_name if not alias_token.startswith('"') else qident(alias_name)
        leaf_token = object_token.split(".")[-1]
        object_name = leaf_token.strip('"')
        reference_map.setdefault(
            object_name.upper(),
            qident(object_name) if leaf_token.startswith('"') else object_name,
        )
    for match in ALIAS_FROM_JOIN_PATTERN.finditer(sql):
        alias = match.group(1).strip('"')
        if alias and alias.upper() not in keywords:
            reference_map[alias.upper()] = alias
    for match in SUBQUERY_ALIAS_PATTERN.finditer(sql):
        alias = match.group(1).strip('"')
        if alias:
            reference_map[alias.upper()] = alias

    def repl(match: re.Match[str]) -> str:
        alias = match.group(1)
        column = match.group(2)
        actual_ref = reference_map.get(alias.upper())
        if actual_ref:
            return f'{actual_ref}."{column}"'
        return match.group(0)

    normalized = re.sub(
        r'"([A-Za-z_][A-Za-z0-9_]*)"\."([A-Za-z_][A-Za-z0-9_]*)"',
        repl,
        sql,
    )
    for upper_alias, actual_ref in reference_map.items():
        normalized = re.sub(
            rf'(?<!")\b{re.escape(upper_alias)}\."([A-Za-z_][A-Za-z0-9_]*)"',
            lambda m: f'{actual_ref}."{m.group(1)}"',
            normalized,
            flags=re.I,
        )
        normalized = re.sub(
            rf'(?<!")\b{re.escape(upper_alias)}\.([A-Za-z_][A-Za-z0-9_]*)\b(?!")(?!(\s*\())',
            lambda m: f'{actual_ref}."{m.group(1)}"',
            normalized,
            flags=re.I,
        )
    return normalized


def has_top_level_comma(text: str) -> bool:
    depth = 0
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            if in_string and i + 1 < len(text) and text[i + 1] == "'":
                i += 2
                continue
            in_string = not in_string
        elif not in_string:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                return True
        i += 1
    return False


def replace_function_calls(sql: str, func_name: str, formatter) -> str:
    pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(", re.I)
    parts: list[str] = []
    cursor = 0
    while True:
        match = pattern.search(sql, cursor)
        if not match:
            parts.append(sql[cursor:])
            break
        open_idx = sql.find("(", match.start())
        depth = 1
        in_string = False
        i = open_idx + 1
        while i < len(sql) and depth > 0:
            ch = sql[i]
            if ch == "'":
                if in_string and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                    continue
                in_string = not in_string
            elif not in_string:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            i += 1
        if depth != 0:
            parts.append(sql[cursor:])
            break
        arg = sql[open_idx + 1 : i - 1]
        replacement = formatter(arg)
        if replacement is None:
            parts.append(sql[cursor:i])
        else:
            parts.append(sql[cursor:match.start()])
            parts.append(replacement)
        cursor = i
    return "".join(parts)


def apply_view_specific_fixes(schema: str, view_name: str, sql: str) -> str:
    fixed = sql
    fixed = re.sub(
        r"''\s+AS\s+\"([A-Za-z0-9_]*Date[A-Za-z0-9_]*)\"",
        r'NULL::timestamp without time zone AS "\1"',
        fixed,
        flags=re.I,
    )

    if view_name == "AgentAll_BOMLIST":
        fixed = fixed.replace('"AgentAll_BOM_LIST"(1)', f'{qident(schema)}."AgentAll_BOM_LIST"')

    if view_name in {"AgentAll_SalesRevenueCost", "MTC_VW_AI_SalesRevenueCost"}:
        fixed = re.sub(
            r'([A-Za-z_][A-Za-z0-9_]*)\."DocNum"\s*=\s*T20\."BASE_REF"',
            r'CAST(\1."DocNum" AS text) = T20."BASE_REF"',
            fixed,
        )

    if view_name == "B1_CashFlowForecastRecurringTrasactionView":
        fixed = re.sub(
            r'T0\."ObjType"\s*=\s*ABS\(CAST\(([^)]+?)\s+AS\s+numeric\)\)',
            r'T0."ObjType" = CAST(ABS(CAST(\1 AS numeric)) AS text)',
            fixed,
            flags=re.I,
        )

    if view_name == "B1_AllDocumentsView_VAWG":
        fixed = re.sub(
            r'SELECT\s+T0\."ObjType"\s*,',
            'SELECT CAST(T0."ObjType" AS text) AS "ObjType",',
            fixed,
            flags=re.I,
        )
        fixed = re.sub(
            r'SELECT\s+(-?\d+)\s+AS\s+"ObjType"',
            r'SELECT CAST(\1 AS text) AS "ObjType"',
            fixed,
            flags=re.I,
        )

    if view_name == "B1_ProductionContributionView":
        fixed = re.sub(
            r'([A-Za-z0-9_\."]+)\s+AS\s+"TransType"',
            r'CAST(\1 AS text) AS "TransType"',
            fixed,
        )

    if view_name == "B1_VatPerPaymentMeansView":
        fixed = re.sub(
            r'-1\s+AS\s+"PayObjType"',
            'CAST(-1 AS text) AS "PayObjType"',
            fixed,
            flags=re.I,
        )

    if view_name in {"AgentAll_OWORCost", "MTC_VW_AI_OWORCost"}:
        fixed = re.sub(
            r'"TransType"\s*=\s*(59|60|202)\b',
            lambda m: f'"TransType" = \'{m.group(1)}\'',
            fixed,
        )

    return fixed


def relation_exists(dsn: str, schema: str, relation_name: str) -> bool:
    sql = (
        "SELECT 1 FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        f"WHERE n.nspname = {quote_literal(schema)} AND c.relname = {quote_literal(relation_name)} "
        "LIMIT 1;"
    )
    return psql_query_scalar(dsn, sql).strip() == "1"


def create_compatibility_objects(dsn: str, schema: str) -> None:
    compatibility_sql: dict[str, str] = {
        "AgentAll_PROCESS_MONITOR_LOG": f"""
            CREATE OR REPLACE VIEW {qident(schema)}."AgentAll_PROCESS_MONITOR_LOG" AS
            SELECT
                NULL::integer AS "ID",
                NULL::text AS "ObjectType",
                NULL::text AS "ObjectNo",
                NULL::text AS "ObjectStatus",
                NULL::text AS "MessageReferenceCode",
                NULL::text AS "MessageType",
                NULL::timestamp without time zone AS "MessageSendTime",
                NULL::text AS "NotifyMethod",
                NULL::text AS "NotifyToName",
                NULL::text AS "NotifyToContact"
            WHERE FALSE;
        """,
        "AgentAll_USRPERM": f"""
            CREATE OR REPLACE VIEW {qident(schema)}."AgentAll_USRPERM" AS
            SELECT
                NULL::integer AS "ID",
                NULL::text AS "USER_CODE",
                NULL::text AS "VIEW_NAME",
                NULL::text AS "USER_PERMISSION"
            WHERE FALSE;
        """,
        "AgentAll_BOM_LIST": f"""
            CREATE OR REPLACE VIEW {qident(schema)}."AgentAll_BOM_LIST" AS
            WITH RECURSIVE bom AS (
                SELECT
                    root."Code" AS "ProductCode",
                    root."Name" AS "ProductName",
                    child."Code" AS "ItemCode",
                    COALESCE(item."ItemName", child."ItemName", child."Code") AS "ItemName",
                    4 AS "Type",
                    'Item'::text AS "TreeTypeName",
                    child."Warehouse" AS "WhsCode",
                    child."IssueMthd" AS "IssueMthd",
                    child."Quantity" AS "Quantity",
                    COALESCE(child."AddQuantit", 0) AS "AddQuantity",
                    root."TreeType" AS "TreeType",
                    child."Father" AS "Father",
                    ARRAY[root."Code", child."Code"]::text[] AS path
                FROM {qident(schema)}."OITT" root
                JOIN {qident(schema)}."ITT1" child ON child."Father" = root."Code"
                LEFT JOIN {qident(schema)}."OITM" item ON item."ItemCode" = child."Code"
                WHERE child."Type" = 4

                UNION ALL

                SELECT
                    bom."ProductCode",
                    bom."ProductName",
                    child."Code" AS "ItemCode",
                    COALESCE(item."ItemName", child."ItemName", child."Code") AS "ItemName",
                    4 AS "Type",
                    bom."TreeTypeName",
                    child."Warehouse" AS "WhsCode",
                    child."IssueMthd" AS "IssueMthd",
                    child."Quantity" AS "Quantity",
                    COALESCE(child."AddQuantit", 0) AS "AddQuantity",
                    bom."TreeType",
                    child."Father" AS "Father",
                    bom.path || child."Code"
                FROM bom
                JOIN {qident(schema)}."ITT1" child ON child."Father" = bom."ItemCode"
                LEFT JOIN {qident(schema)}."OITM" item ON item."ItemCode" = child."Code"
                WHERE child."Type" = 4
                  AND NOT child."Code" = ANY(bom.path)
            )
            SELECT
                bom."ProductCode",
                bom."ProductName",
                bom."ItemCode",
                bom."ItemName",
                bom."Type",
                CASE bom."TreeType"
                    WHEN 'P' THEN 'Production'
                    WHEN 'S' THEN 'Sales'
                    WHEN 'T' THEN 'Template'
                    ELSE bom."TreeType"
                END AS "TreeTypeName",
                bom."WhsCode",
                bom."IssueMthd",
                bom."Quantity",
                bom."AddQuantity",
                bom."TreeType",
                bom."Father"
            FROM bom
            WHERE NOT EXISTS (
                SELECT 1
                FROM {qident(schema)}."ITT1" child
                WHERE child."Father" = bom."ItemCode"
                  AND child."Type" = 4
            );
        """,
    }
    for object_name, sql in compatibility_sql.items():
        if relation_exists(dsn, schema, object_name):
            continue
        psql_exec(dsn, sql)
        create_lowercase_alias(dsn, schema, object_name)


def convert_view_sql(schema: str, view_name: str, body: str) -> str:
    converted = body
    converted = re.sub(r"\bN'([^']*)'", lambda m: quote_literal(m.group(1)), converted)
    converted = re.sub(r'IFNULL\s*\(', "COALESCE(", converted, flags=re.I)
    converted = re.sub(
        r'COALESCE\(\s*("[^"]+"|"[^"]+"\."[^"]+"|[A-Za-z_][A-Za-z0-9_\."]*)\s*,\s*\'\'\s*\)',
        lambda m: f"COALESCE(CAST({m.group(1)} AS text), '')",
        converted,
        flags=re.I,
    )
    converted = re.sub(r"CAST\((.*?)\s+AS\s+NVARCHAR\(\d+\)\)", r"CAST(\1 AS text)", converted, flags=re.I | re.S)
    converted = re.sub(r"CAST\((.*?)\s+AS\s+NVARCHAR\)", r"CAST(\1 AS text)", converted, flags=re.I | re.S)
    converted = re.sub(r"\bSELECT\s+TOP\s+1\b", "SELECT", converted, flags=re.I)
    converted = replace_function_calls(
        converted,
        "YEAR",
        lambda arg: None if has_top_level_comma(arg) else f"EXTRACT(YEAR FROM {arg.strip()})",
    )
    converted = replace_function_calls(
        converted,
        "MONTH",
        lambda arg: None if has_top_level_comma(arg) else f"EXTRACT(MONTH FROM {arg.strip()})",
    )
    converted = replace_function_calls(
        converted,
        "DAY",
        lambda arg: None if has_top_level_comma(arg) else f"EXTRACT(DAY FROM {arg.strip()})",
    )
    converted = replace_function_calls(
        converted,
        "TO_INT",
        lambda arg: None if has_top_level_comma(arg) else f"CAST({arg.strip()} AS integer)",
    )
    converted = replace_function_calls(
        converted,
        "TO_NUMBER",
        lambda arg: None if has_top_level_comma(arg) else f"CAST({arg.strip()} AS numeric)",
    )
    converted = replace_function_calls(
        converted,
        "TO_CHAR",
        lambda arg: None if has_top_level_comma(arg) else f"CAST({arg.strip()} AS text)",
    )
    converted = replace_function_calls(
        converted,
        "ABS",
        lambda arg: None if has_top_level_comma(arg) else f"ABS(CAST({arg.strip()} AS numeric))",
    )
    converted = replace_function_calls(
        converted,
        "LENGTH",
        lambda arg: None if has_top_level_comma(arg) else f"LENGTH(CAST({arg.strip()} AS text))",
    )
    converted = convert_days_between(converted)
    converted = strip_security_joins(converted)
    converted = normalize_quoted_alias_references(converted)
    converted = converted.replace('"SBODEMOUS".', f'{qident(schema)}.')
    converted = re.sub(r"\bSBODEMOUS\.", f"{schema}.", converted)
    converted = apply_view_specific_fixes(schema, view_name, converted)
    sql = (
        f"SET search_path TO {qident(schema)}, public; "
        f"CREATE OR REPLACE VIEW {qident(schema)}.{qident(view_name)} AS\n"
        f"{converted.strip()};"
    )
    return sql


def stream_copy_into_table(dsn: str, schema: str, table_name: str, csv_stream: io.BufferedReader) -> None:
    copy_sql = (
        f"COPY {qident(schema)}.{qident(table_name)} "
        f"FROM STDIN WITH (FORMAT csv, HEADER true)"
    )
    proc = subprocess.Popen(
        ["psql", dsn, "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", copy_sql],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        while True:
            chunk = csv_stream.read(1024 * 1024)
            if not chunk:
                break
            proc.stdin.write(chunk)
    finally:
        proc.stdin.close()
    stdout = proc.stdout.read() if proc.stdout is not None else b""
    stderr = proc.stderr.read() if proc.stderr is not None else b""
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8") or stdout.decode("utf-8"))


def relax_not_null_and_retry(
    dsn: str,
    schema: str,
    table_name: str,
    error_text: str,
    csv_member_name: str,
    backup_tar_path: str,
) -> bool:
    match = re.search(r'null value in column "([^"]+)"', error_text, flags=re.I)
    if not match:
        return False
    column_name = match.group(1)
    psql_exec(
        dsn,
        f"ALTER TABLE {qident(schema)}.{qident(table_name)} "
        f"ALTER COLUMN {qident(column_name)} DROP NOT NULL;",
    )
    with tarfile.open(backup_tar_path, "r:gz") as tf:
        with tf.extractfile(csv_member_name) as raw:
            assert raw is not None
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                stream_copy_into_table(dsn, schema, table_name, gz)
    return True


def create_lowercase_alias(dsn: str, schema: str, object_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", object_name):
        return
    alias_name = object_name.lower()
    if alias_name == object_name:
        return
    sql = (
        f"CREATE OR REPLACE VIEW {qident(schema)}.{alias_name} AS "
        f"SELECT * FROM {qident(schema)}.{qident(object_name)};"
    )
    psql_exec(dsn, sql)


def create_import_log_table(dsn: str, schema: str) -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {qident(schema)}.import_run_log (
        run_id text PRIMARY KEY,
        started_at timestamptz NOT NULL,
        finished_at timestamptz NOT NULL,
        report_path text NOT NULL,
        table_success integer NOT NULL,
        table_fail integer NOT NULL,
        import_success integer NOT NULL,
        import_fail integer NOT NULL,
        view_success integer NOT NULL,
        view_fail integer NOT NULL,
        summary jsonb NOT NULL
    );
    """
    psql_exec(dsn, sql)


def insert_import_log(dsn: str, schema: str, report_path: str, report: dict[str, Any]) -> None:
    summary = {
        "table_success": report["summary"]["tables_created"],
        "table_fail": report["summary"]["table_failures"],
        "import_success": report["summary"]["data_imported"],
        "import_fail": report["summary"]["data_failures"],
        "view_success": report["summary"]["views_created"],
        "view_fail": report["summary"]["view_failures"],
    }
    summary_json = json.dumps(summary).replace("'", "''")
    sql = f"""
    INSERT INTO {qident(schema)}.import_run_log (
        run_id, started_at, finished_at, report_path, table_success, table_fail,
        import_success, import_fail, view_success, view_fail, summary
    ) VALUES (
        {quote_literal(report['run_id'])},
        {quote_literal(report['started_at'])}::timestamptz,
        {quote_literal(report['finished_at'])}::timestamptz,
        {quote_literal(report_path)},
        {summary['table_success']},
        {summary['table_fail']},
        {summary['import_success']},
        {summary['import_fail']},
        {summary['view_success']},
        {summary['view_fail']},
        {quote_literal(summary_json)}::jsonb
    );
    """
    psql_exec(dsn, sql)


def validate_counts(dsn: str, schema: str, manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected = {row["TABLE_NAME"]: int(row["RECORD_COUNT"] or "0") for row in manifest_rows}
    for table_name in ["CPRF", "AgentAll_TRANS_LOG", "OINV", "OITM"]:
        try:
            actual = psql_query_scalar(
                dsn,
                f"SELECT COUNT(*) FROM {qident(schema)}.{qident(table_name)};",
            )
            checks[table_name] = {
                "expected": expected.get(table_name),
                "actual": int(actual),
                "matched": int(actual) == expected.get(table_name),
            }
        except Exception as exc:  # noqa: BLE001
            checks[table_name] = {"error": str(exc)}
    return checks


def validate_views(dsn: str, schema: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for view_name in ["MTC_VW_AI_OINV", "MTC_VW_AI_STOCK", "MTC_VW_AI_SalesRevenueCost"]:
        try:
            psql_exec(
                dsn,
                f"SELECT 1 FROM {qident(schema)}.{qident(view_name)} LIMIT 1;",
                capture=True,
            )
            checks[view_name] = {"ok": True}
        except Exception as exc:  # noqa: BLE001
            checks[view_name] = {"ok": False, "error": str(exc)}
    return checks


def load_existing_views(dsn: str, schema: str) -> set[str]:
    sql = (
        "SELECT table_name FROM information_schema.views "
        f"WHERE table_schema = {quote_literal(schema)}"
    )
    rows = psql_query_scalar(dsn, sql)
    if not rows:
        return set()
    return {line.strip() for line in rows.splitlines() if line.strip()}


def main() -> int:
    args = parse_args()
    run_id = str(uuid.uuid4())
    started_at = utc_now()

    target_schema = args.target_schema
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "backup_tar": os.path.abspath(args.backup_tar),
        "ddl_part1": os.path.abspath(args.ddl_part1),
        "ddl_part2": os.path.abspath(args.ddl_part2),
        "pg_dsn": args.pg_dsn,
        "target_schema": target_schema,
        "scope": args.scope,
        "tables": {
            "created_from_ddl": [],
            "created_from_csv_fallback": [],
            "missing_ddl_and_csv": [],
            "create_failures": [],
            "data_imported": [],
            "data_import_failures": [],
        },
        "views": {
            "created": [],
            "failures": [],
            "skipped_due_to_no_progress": [],
        },
        "validation": {},
        "summary": {},
    }

    admin_dsn, dbname = ensure_database(args.pg_dsn)
    recreate_database(admin_dsn, dbname)
    create_schema(args.pg_dsn, target_schema)

    ddl_text = Path(args.ddl_part1).read_text(encoding="utf-8", errors="ignore")
    table_definitions = parse_table_definitions(ddl_text)
    view_definitions = parse_view_definitions(ddl_text)

    with tarfile.open(args.backup_tar, "r:gz") as tf:
        manifest_rows = load_manifest(tf)
        csv_members = load_csv_members(tf)

        for row in manifest_rows:
            table_name = row["TABLE_NAME"]
            ddl_body = table_definitions.get(table_name)
            csv_member = csv_members.get(table_name)
            try:
                if ddl_body:
                    sql = build_table_sql(target_schema, table_name, ddl_body)
                    psql_exec(args.pg_dsn, sql)
                    create_lowercase_alias(args.pg_dsn, target_schema, table_name)
                    report["tables"]["created_from_ddl"].append(table_name)
                elif csv_member:
                    header = load_csv_header(tf, csv_member)
                    sql = build_fallback_table_sql(target_schema, table_name, header)
                    psql_exec(args.pg_dsn, sql)
                    create_lowercase_alias(args.pg_dsn, target_schema, table_name)
                    report["tables"]["created_from_csv_fallback"].append(table_name)
                else:
                    report["tables"]["missing_ddl_and_csv"].append(table_name)
            except Exception as exc:  # noqa: BLE001
                report["tables"]["create_failures"].append(
                    {"table": table_name, "error": str(exc)}
                )

        for table_name, member_name in csv_members.items():
            try:
                with tf.extractfile(member_name) as raw:
                    assert raw is not None
                    with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                        stream_copy_into_table(args.pg_dsn, target_schema, table_name, gz)
                report["tables"]["data_imported"].append(table_name)
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                recovered = False
                try:
                    recovered = relax_not_null_and_retry(
                        args.pg_dsn,
                        target_schema,
                        table_name,
                        error_text,
                        member_name,
                        args.backup_tar,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    error_text = f"{error_text}\nRETRY: {retry_exc}"
                if recovered:
                    report["tables"]["data_imported"].append(table_name)
                else:
                    report["tables"]["data_import_failures"].append(
                        {"table": table_name, "error": error_text}
                    )

    pending_views = dict(view_definitions)
    create_compatibility_objects(args.pg_dsn, target_schema)
    while pending_views:
        progressed = False
        batch_failures: list[tuple[str, str]] = []
        for view_name, body in list(pending_views.items()):
            sql = convert_view_sql(target_schema, view_name, body)
            try:
                psql_exec(args.pg_dsn, sql)
                create_lowercase_alias(args.pg_dsn, target_schema, view_name)
                report["views"]["created"].append(view_name)
                del pending_views[view_name]
                progressed = True
            except Exception as exc:  # noqa: BLE001
                batch_failures.append((view_name, str(exc)))
        if progressed:
            continue
        for view_name, error in batch_failures:
            report["views"]["failures"].append({"view": view_name, "error": error})
        report["views"]["skipped_due_to_no_progress"] = list(pending_views)
        break

    report["validation"]["row_counts"] = validate_counts(
        args.pg_dsn, target_schema, manifest_rows
    )
    report["validation"]["views"] = validate_views(args.pg_dsn, target_schema)

    finished_at = utc_now()
    report["finished_at"] = finished_at.isoformat()
    report["summary"] = {
        "database": dbname,
        "tables_manifest": len(manifest_rows),
        "tables_created": len(report["tables"]["created_from_ddl"])
        + len(report["tables"]["created_from_csv_fallback"]),
        "table_failures": len(report["tables"]["create_failures"])
        + len(report["tables"]["missing_ddl_and_csv"]),
        "data_files": len(report["tables"]["data_imported"])
        + len(report["tables"]["data_import_failures"]),
        "data_imported": len(report["tables"]["data_imported"]),
        "data_failures": len(report["tables"]["data_import_failures"]),
        "views_defined": len(view_definitions),
        "views_created": len(report["views"]["created"]),
        "view_failures": len(report["views"]["failures"])
        + len(report["views"]["skipped_due_to_no_progress"]),
        "elapsed_seconds": round((finished_at - started_at).total_seconds(), 2),
        "csv_fallback_tables_required": sorted(CSV_FALLBACK_TABLES),
    }

    report_dir = Path(args.backup_tar).resolve().parent
    report_path = report_dir / f"pg-import-report-{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    create_import_log_table(args.pg_dsn, target_schema)
    insert_import_log(args.pg_dsn, target_schema, str(report_path), report)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
