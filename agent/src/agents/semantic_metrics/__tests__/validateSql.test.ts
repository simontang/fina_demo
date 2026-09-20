import { describe, expect, it } from "@jest/globals";
import { effectiveDatasourceScope, validateSql } from "../tools/types";

describe("validateSql", () => {
  it("allows a simple SELECT", () => {
    expect(validateSql("select * from hankel_distr_sell_in")).toEqual({ ok: true });
  });

  it("allows a SELECT with trailing semicolon", () => {
    expect(validateSql("select count(*) from t;")).toEqual({ ok: true });
  });

  it("allows a WITH (CTE) query", () => {
    expect(validateSql("WITH x AS (SELECT 1) SELECT * FROM x")).toEqual({ ok: true });
  });

  it("allows comments before SELECT", () => {
    expect(validateSql("-- header\nselect 1")).toEqual({ ok: true });
  });

  it("rejects empty SQL", () => {
    expect(validateSql("   ").ok).toBe(false);
  });

  it.each(["insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke", "merge", "replace", "upsert"])(
    "rejects %s appearing mid-statement",
    (keyword) => {
      expect(validateSql(`WITH x AS (SELECT 1) SELECT * FROM x WHERE ${keyword} = 1`).ok).toBe(false);
    },
  );

  it("rejects SELECT INTO", () => {
    expect(validateSql("select * into backup_t from t").ok).toBe(false);
  });

  it("allows identifiers that merely contain keyword substrings", () => {
    expect(validateSql("select updated_at, created_by from replacements")).toEqual({ ok: true });
    expect(validateSql("select * from deleted_at_log")).toEqual({ ok: true });
  });

  it("rejects DML inside a CTE", () => {
    expect(validateSql("WITH x AS (DELETE FROM t) SELECT * FROM x").ok).toBe(false);
  });

  it("rejects multiple statements", () => {
    expect(validateSql("select 1; select 2").ok).toBe(false);
  });

  it("rejects a non-SELECT leading keyword", () => {
    expect(validateSql("explain select 1").ok).toBe(false);
  });
});

describe("effectiveDatasourceScope", () => {
  it("is unrestricted when the connection selects no resources", () => {
    expect(effectiveDatasourceScope([])).toEqual({ unrestricted: true, ids: [] });
  });

  it("uses the connection's selected resources when present", () => {
    expect(effectiveDatasourceScope([15, 16])).toEqual({ unrestricted: false, ids: [15, 16] });
  });
});
